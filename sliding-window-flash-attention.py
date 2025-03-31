import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    sliding_window_size,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handeled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal
        lo = max(0, (block_index_q * BLOCK_SIZE_Q) - sliding_window_size * BLOCK_SIZE_Q)
        hi = block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # used only for the block in which there is transition between non-masked and masked keys
        lo = block_index_q * BLOCK_SIZE_Q
        hi = min((block_index_q + 1) * BLOCK_SIZE_Q, SEQ_LEN)
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Only used for non-casual attention with sliding window 
        lo = max(0, (block_index_q * BLOCK_SIZE_Q) - sliding_window_size * BLOCK_SIZE_Q) # Apply sliding window
        hi = min((block_index_q + 1) * BLOCK_SIZE_Q + sliding_window_size * BLOCK_SIZE_Q, SEQ_LEN) # Apply sliding window

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # ---- compute qk ------------
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        # Apply sliding wwindow mask
        if STAGE == 2:
            # Casual mask within the sliding window
            q_pos = offs_q[:, None] 
            k_pos = start_kv + offs_kv[None, :]
            casual_mask = q_pos >= k_pos
            window_mask = tl.abs(q_pos - k_pos) <= (sliding_window_size * BLOCK_SIZE_Q)
            # Apply the mask to the QK_block
            combined_mask = casual_mask & window_mask
            QK_block = QK_block * softmax_scale + tl.where(combined_mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1))
            QK_block -= m_ij[:, None]
        else:
            # Apply just the sliding window mask for non-casual parts
            q_pos = offs_q[:, None]
            k_pos = start_kv + offs_kv[None, :]
            window_mask = tl.abs(q_pos - k_pos) <= (sliding_window_size * BLOCK_SIZE_Q)
            QK_block = QK_block * softmax_scale + tl.where(window_mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1))
            QK_block -= m_ij[:, None]

        """
        Block Processing Flow:

        Block 1 (K0-3):                 Block 2 (K4-7):                 Final m_ij
        Q0: max=1.63 ─────────────────► Q0: max=2.04 ─────────────────► max(1.63, 2.04) = 2.04
        Q1: max=2.04 ─────────────────► Q1: max=2.45 ─────────────────► max(2.04, 2.45) = 2.45
        Q2: max=2.45 ─────────────────► Q2: max=2.04 ─────────────────► max(2.45, 2.04) = 2.45
        Q3: max=2.04 ─────────────────► Q3: max=2.45 ─────────────────► max(2.04, 2.45) = 2.45

        m_i after Block 1              m_i after Block 2
        [1.63, 2.04, 2.45, 2.04] ──► [2.04, 2.45, 2.45, 2.45]

        Process:
        1. Each query computes max over K0-K3
        2. m_i stores these maximums
        3. Each query computes max over K4-K7
        4. Final m_ij takes maximum between stored m_i and new block max
        """

        # Compute the exponential of each dot product, so we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, axis=1)

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij
 
        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None] 
        O_block = tl.dot(P_block, V_block, out=O_block)

        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_SIZE_KV, 0))
    return O_block, l_i, m_i

@triton.jit
def _attn_bwd_process(
    O,
    dO,
    D, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN]
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr, # suppose 4
    HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0) # What is the block of vectors of O we are working with ?
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q) # This is the offset of the block of O we are working with
    # offs_q =  33, 34, 35, 36 if 33 other is mamanaged by other programs
    index_batch_head = tl.program_id(1) # What is the batch and head inside of each batch we are working with ?
    offs_dim = tl.arange(0, HEAD_DIM) # This is the offset of the head_dim we are working with

    # Load a single block of BLOCK_SIZE_O rows of O
    O_block = tl.load( # O [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM 
        + offs_dim[None, :] 
    ) # Shape : (BLOCK_SIZE_Q, HEAD_DIM)

    # Load a single block of BLOCK_SIZE_O rows of dO
    dO_block = tl.load(
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ).to(tl.float32) # Shape : (BLOCK_SIZE_Q, HEAD_DIM)

    # Compute Di block 
    D_block = tl.sum(dO_block * O_block, axis=1) # Shape : (BLOCK_SIZE_Q, )
    # Store in the Di block
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)

@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)  # Right Batch and right head 
    index_batch = index_batch_head // NUM_HEADS  # Select the right Batch 
    index_head = index_batch_head % NUM_HEADS # Select the right Head
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64 # Enter's the right batch and right head
    ) 
    # This is the offset that allows us to select the right sequence given the batch and the head
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64) 

    # Make sure the pointers are in the right place w.r.t the batch and head 
    # The reason we don't acess the blocks through make_block_ptr is bcs we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointer's are in right place w.r.t the batch, head and sequence
    M += offset_batch_head_seq # M [BATCH_SIZE, NUM_HEADS, SEQ_LEN]
    D += offset_batch_head_seq # D [BATCH_SIZE, NUM_HEADS, SEQ_LEN]

    # load sclaes 
    offs_dim = tl.arange(0, HEAD_DIM) 

    index_block_kv = tl.program_id(0) 
    start_kv = index_block_kv * BLOCK_KV

    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim # Shape : [BLOCK_SIZE_KV, HEAD_DIM]
    ) # Example K [0, 0, start_kv: start_kv + BLOCK_KV, 0: HEAD_DIM]
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim # Shape : [BLOCK_SIZE_KV, HEAD_DIM]
    ) # Example V [0, 0, start_kv: start_kv + BLOCK_KV, 0: HEAD_DIM]

    offs_q = tl.arange(0, BLOCK_Q) # For each block of Queries how many vectors we should load

    # offs_q = 0, 1*128, 2*128, 3*128, 4*128, 5*128, 6*128, 7*128
    # This is for acessing Q and dO
    # For now let's assure the head_dim = 4
    # 0  (0, 1, 2, 3) = (0, 1, 2, 3)
    # 1*4 (4, 5, 6, 7) = (4, 5, 6, 7)
    # 2*4 (8, 9, 10, 11) = (8, 9, 10, 11)
    # 3*4 (12, 13, 14, 15) = (12, 13, 14, 15)
    # 
    """
    We access the Q as a transposed array, so that's why we treat offs_q as a coulmn vector and offs_dim as a row vector
    This is equivalent to doing:
    """
    # q_ptrs = Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    # qT_ptrs = tl.trans(q_ptrs)
    # We point to the first BLOCK_Q rows of the Q for both the qT and dO pointers, inside of the for loop we will move forward by BLOCK_Q rows at each iteration
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None ] * stride_dim # Shape : [BLOCK_Q, HEAD_DIM]
    # qT_ptrs = [
    # [  0,   4,   8,  12],  # Head Dim 0
    # [  1,   5,   9,  13],  # Head Dim 1
    # [  2,   6,  10,  14],  # Head Dim 2
    # [  3,   7,  11,  15],  # Head Dim 3
    # ]
    dO_ptrs = dO + offs_q[None, :] * stride_seq + offs_dim[None, :] * stride_dim # Shape : [BLOCK_Q, HEAD_DIM]
    # dO_ptrs = [
    # [  0,   1,   2,   3],  # Sequence 0
    # [  4,   5,   6,   7],  # Sequence 1
    # [  8,   9,  10,  11],  # Sequence 2
    # [ 12,  13,  14,  15],  # Sequence 3
    # ]
    
    # Iterate over the sequence dimention of the query
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q # How many blocks of Q we have ? 1024 / 32 = 32
    for blk_index in range(num_steps):
        # Load the block of Q
        qT_block = tl.load(qT_ptrs) # Shape : [BLOCK_Q, HEAD_DIM]
        # Load the logsumexp values for the queries in the current block
        offs_q = curr_q + tl.arange(0, BLOCK_Q) # This is the offset of the block of Q we are working with
        m = tl.load(M + offs_q)

        # This gives us (QK^T)^T = (K^T)^T(Q^T) = K(Q^T) = S^T
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block) # Shape : [BLOCK_SIZE_KV, BLOCK_Q]
        # We apply the softmax by using the logsumexp trick
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            # Autoaggressive masking.
            # mask is True for all the values that DO NOT NEED TO BE MASKED
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None] # This is the mask for the autoaggressive part
            ) # Shape : [BLOCK_SIZE_KV, BLOCK_Q]
            # Replace all the masked value with 0
            # In this case we do not need to mask -Inf before applying the softmax since we have already computed the normalization factors (stored in "m")
            P_T_block = tl.where(mask_block, P_T_block, 0.0)
        
        dO_block = tl.load(dO_ptrs) 
        # According to the formula: dV_new = dV_old + P^T x dO, where x is the matrix multiplication
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block) # Shape : [BLOCK_SIZE_KV, HEAD_DIM]

        # Delta = rowsum(O * dO) where * is the elementwise product
        Di = tl.load(D + offs_q) # Shape : [BLOCK_Q, ]

        # dP = dO x V^T, so dP^T = V x dO^T
        # Where x is the matrix multiplication
        dpT_block = tl.dot(V_block, tl.tans(dO_block)).to(tl.float32)

        # We know that dS = P * (dP - Delta), so dS^T = P^T * (dP^T - Delta^T)

        dS_T_block = P_T_block * (dPT_block - Di[None, :]) # Shape : [BLOCK_SIZE_KV, BLOCK_Q]
        dS_T_block = dS_T_block.to(tl.float16) # Convert to float16 

        # According to the formula on the paper dK_new = dK_old + dS^T x Q
        dK_block += softmax_scale * tl.dot(ds_T_block, tl.trans(qT_block)) # Shape : [BLOCK_SIZE_KV, BLOCK_Q]





@triton.jit
def _attn_fwd(
    Q, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    K, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    V, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    softmax_scale,
    sliding_window_size, # How many tokens to the left and right of the current token to consider
    M, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN]
    O, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # This indicates which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # This indicates which head and batch to process. Each program is associated with a single head of single batch element
    index_batch_head = tl.program_id(1)
    
    # This indicates which batch this program is associated with (Each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS # Select the right Batch
    
    # This indicate the position of the head in the batch 
    index_head = index_batch_head % NUM_HEADS # Select the right Head

    # This allows to get the (SEQ_LEN, HEAD_DIM) block of Q, K, V by selecting indexing it by batch and head
    qkv_offset = (
        index_batch.to(tl.int32) * stride_Q_batch # Q[index_batch * stride_Q_batch, :, :, :]
        + index_head.to(tl.int32) * stride_Q_head # Q[index_batch * stride_Q_batch + index_head * stride_Q_head, :, :]
    )

    # We are in Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q :, :]
    Q_block_ptr = tl.make_block_ptr( # Currently pointing the perticular program to be working with 
        base = Q + qkv_offset, # Q[index_batch, index_head, :, :]
        shape = (SEQ_LEN, HEAD_DIM), 
        strides = (stride_Q_seq, stride_Q_dim), 
        offsets = (block_index_q * BLOCK_SIZE_Q, 0), # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q :, :]
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM), # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q :, :]
        order = (1,0), 
    )

    """
        Q tensor:
    [
    [  # Batch 0
        [  # Head 0
        [Q0, Q1, Q2, Q3],  # Sequence 0
        [Q4, Q5, Q6, Q7],  # Sequence 1
        [Q8, Q9, Q10, Q11],  # Sequence 2
        [Q12, Q13, Q14, Q15],  # Sequence 3
        ],
        [  # Head 1
        ...
        ]
    ],
    [  # Batch 1
        [  # Head 0
        [Q32, Q33, Q34, Q35],  # Sequence 0
        [Q36, Q37, Q38, Q39],  # Sequence 1
        [Q40, Q41, Q42, Q43],  # Sequence 2
        [Q44, Q45, Q46, Q47],  # Sequence 3
        ],
        [  # Head 1
        ...
        ]
    ]
    ]

    Q_block_ptr points to:
    Batch 1, Head 0, Sequence 4 to 7
    [
    [Q48, Q49, Q50, Q51],  # Sequence 4
    [Q52, Q53, Q54, Q55],  # Sequence 5
    [Q56, Q57, Q58, Q59],  # Sequence 6
    [Q60, Q61, Q62, Q63]   # Sequence 7
    ]
    """

    # We are in V[index_batch, index_head, :, :]
    V_block_ptr = tl.make_block_ptr(
        base = V + qkv_offset, 
        shape = (SEQ_LEN, HEAD_DIM), 
        strides = (stride_V_seq, stride_V_dim),
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM), 
        order = (1,0),
    )

    # We are in  K[index_batch, index_head, :, :]
    """
    Actually it won't be selecting `everything that is inside` but only the number of elements indicated
    by the `block_shape` parameter of each pointer block. You can consider each pointers block to be
    a tensor of pointers with the shape indicated by the param `block_shape`
    """
    K_block_ptr = tl.make_block_ptr(
      base = K + qkv_offset,
      shape=(HEAD_DIM, SEQ_LEN),
      strides=(
        stride_K_dim,
        stride_K_seq,
      ), # We invert the strides w.r.t Q, so we can transpose the matrix
      offsets=(0,0),
      block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
      order=(0,1),
    )

    # In this the selection of the pointer should exactly indicate the right pointer for writing
    # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q :, :]
    O_block_ptr = tl.make_block_ptr(
      base= O + qkv_offset,
      shape=(SEQ_LEN, HEAD_DIM),
      strides=(stride_O_seq, stride_O_dim),
      offsets=(block_index_q * BLOCK_SIZE_Q, 0),
      block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
      order=(1,0),
    )

    # offs_q: the offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # Suppose program=0, block_size=4, Q[0, 1, 2, 3], Suppose program=3, block_size=4, Q[13, 14, 15, 16]
    """Each block of query is made up of block_size_q no of Queries. Each Q is a token and its
    dimention is not all the token but only the part of the head_dim """
    # offs_kv: the offsets for the tokens in the K and V sequences to process
    """We don't skip any values like Q here bcs we are going to multiply the whole K and V with the Q"""
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    # For KV Suppose block_size = 4 -> [0, 1, 2, 3]
    # m_i : the running maximum of each row. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i : the running logsumexp of each row. We have one for each query
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0 # here +1 is to make the log stable
    # acc: the accumilator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)

    # Stage: 3 if casual else 1
    if STAGE == 1 or STAGE == 3:
        # This step runs for non-casual attention or for the blocks to the left of the diagonal in the casual attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            sliding_window_size,  # Pass sliding window size
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the block to the right of the diagonal in the casual attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            sliding_window_size,  # Pass sliding window size
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

        # epilogue: write the output block to the global memory
        m_i += tl.math.log(
            l_i
        ) # This is needed to compute the logsumexp for the backward pass
        O_block = O_block /l_i[:, None] 
        m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, O_block)


class TritionAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, casual, softmax_scale, sliding_window_size=None):
        # Default sliding window size if not provided
        if sliding_window_size is None:
            sliding_window_size = Q.shape[2]  # Default to full sequence length (no windowing)

        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if casual else 2

        grid = lambda args: (
            # Example : SEQ_LEN = 8, BLOCK_SIZE_Q = 4, ceil(SEQ_LEN / BLOCK_SIZE_Q) = 2
            # ceil(SEQ_LEN / BLOCK_SIZE_Q) = How many Blocks of Q we have ?
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), # Which group of Queries are we going to work with ?
            # Example : BATCH_SIZE = 2, BLOCK_SIZE_Q = 4,  BATCH_SIZE * NUM_HEADS = 8, ceil(BATCH_SIZE * NUM_HEADS / BLOCK_SIZE_Q) = 2
            BATCH_SIZE * NUM_HEADS, # Which head of which batch element are we going 
            1, # Z dimension
        )

        # Number of parallel program kernals: (BATCH_SIZE * NUM_HEADS * NUM_BLOCK_Q)
        # (2 * 8 * 4) = 64

        # M is logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        # Select optimal block sizes based on sequence length and sliding window size
        optimal_block_size_q = min(128, SEQ_LEN)
        optimal_block_size_kv = min(HEAD_DIM, 128, sliding_window_size * 32) # Scale based on window size

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            O=O,
            M=M,
            softmax_scale=softmax_scale,
            sliding_window_size=sliding_window_size,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=Q.shape[3],
            BLOCK_SIZE_Q=optimal_block_size_q,
            BLOCK_SIZE_KV=optimal_block_size_kv,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.casual = casual
        ctx.sliding_window_size = sliding_window_size
        return 0
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M) # Shape : (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

        # Compute all the elements Di
        _attn_bwd_process[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)

        stage = 3 if ctx.casual else 1

        # Fix KV and iterate through all the blocks of Q
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch = Q.stride(0),
            stride_head = Q.stride(1),
            stride_seq = Q.stride(2),
            stride_dim = Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        
        

# Comparision with naive PyTorch implementation and Triton implementation
def test_op_sw(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, sliding_window_size, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=torch.device("cuda")
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=torch.device("cuda")
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=torch.device("cuda")
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM ** 0.5)
    dO = torch.rand_like(Q)

    # reference implementation with sliding window
    MASK = torch.ones(SEQ_LEN, SEQ_LEN).to(Q.device)

    # Create sliding window mask
    for i in range(SEQ_LEN):
        for j in range(SEQ_LEN):
            if abs(i - j) > sliding_window_size:
                MASK[i, j] = 0.0
    
    # Apply casual mask if needed
    if causal:
        CASUAL_MASK = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).to(Q.device)
        MASK = MASK * (1 - CASUAL_MASK)

    # Calculate attention
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    P = P.masked_fill(MASK.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    P = torch.nn.functional.softmax(P, dim=-1).half()
    ref_O = torch.matmul(P, V)
    
    # Backward pass
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    
    # Triton implementation with sliding window
    tri_out = TritionAttention.apply(Q, K, V, causal, softmax_scale, sliding_window_size).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None

    # Compare results
    rtol = 0.0
    atol = 1e-2
    print(f"Output max diff: {torch.max(torch.abs(ref_O - tri_out))}")
    assert torch.allclose(ref_O, tri_out, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dV, tri_dV, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dQ, tri_dQ, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dK, tri_dK, rtol=rtol, atol=atol)
    
    return "All tests passed!"

if __name__ == "__main__":
    test_op_sw(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True, sliding_window_size=64)
    print("PASSED")