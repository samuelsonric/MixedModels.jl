struct ChordalWorkspace{T}
    blocks::Vector{AbstractMatrix{T}}
    matrix::SparseMatrixCSC{T, Int}
    indices::Vector{Vector{Int}}
end

_nonzeros(A::Diagonal) = A.diag
_nonzeros(A::Matrix) = vec(A)
_nonzeros(A::UniformBlockDiagonal) = vec(A.data)
_nonzeros(A::BlockedSparse) = nonzeros(A.cscmat)
_nonzeros(A::SparseMatrixCSC) = nonzeros(A)

_nnz(A::Diagonal) = length(A.diag)
_nnz(A::Matrix) = length(A)
_nnz(A::UniformBlockDiagonal) = length(A.data)
_nnz(A::BlockedSparse) = nnz(A.cscmat)
_nnz(A::SparseMatrixCSC) = nnz(A)

_similar(A::Diagonal{T}) where {T} = Diagonal(similar(A.diag))
_similar(A::Matrix{T}) where {T} = similar(A)
_similar(A::UniformBlockDiagonal{T}) where {T} = UniformBlockDiagonal(similar(A.data))
_similar(A::BlockedSparse{T}) where {T} = BlockedSparse(similar(A.cscmat), A.nzasmat, A.colblkptr)
_similar(A::SparseMatrixCSC{T}) where {T} = similar(A)

function _csc(blocks::Vector{<:AbstractMatrix{T}}, reterms::Vector{<:AbstractReMat{T}}) where {T}
    nblkcol = length(reterms) + 1
    nblkptr = length(blocks)
    nfixcol = size(blocks[kp1choose2(nblkcol)], 1)
    ncol = nfixcol
    nptr = 0

    indices = Vector{Vector{Int}}(undef, nblkptr)

    for R in reterms
        ncol += size(R.λ, 1) * nlevs(R)
    end

    for (i, B) in enumerate(blocks)
        nptr += n = _nnz(B)
        indices[i] = Vector{Int}(undef, n)
    end

    colptr = Vector{Int}(undef, ncol + 1)
    rowval = Vector{Int}(undef, nptr)
    nzval = Vector{T}(undef, nptr)
    col = 1; colptr[col] = ptr = 1
    noffcol = 0

    for blkcol in 1:nblkcol
        if blkcol < nblkcol
            R = reterms[blkcol]; nloccol = size(R.λ, 1) * nlevs(R)
        else
            nloccol = nfixcol
        end

        for loccol in 1:nloccol
            noffrow = noffcol

            for blkrow in blkcol:nblkcol
                blkptr = block(blkrow, blkcol)

                ptr = _addcol!(rowval, nzval, indices[blkptr], blocks[blkptr], loccol, noffrow, ptr)

                if blkrow < nblkcol
                    noffrow += size(reterms[blkrow].λ, 1) * nlevs(reterms[blkrow])
                else
                    noffrow += nfixcol
                end
            end

            col += 1; colptr[col] = ptr
        end

        noffcol += nloccol
    end

    S = SparseMatrixCSC(ncol, ncol, colptr, rowval, nzval)
    return S, indices
end

function _addcol!(
        rowval::Vector{Int},
        nzval::Vector{T},
        ind::Vector{Int},
        A::Diagonal{T},
        loccol::Int,
        noffrow::Int,
        ptr::Int,
    ) where {T}
    rowval[ptr] = noffrow + loccol
    nzval[ptr] = A.diag[loccol]
    ind[loccol] = ptr
    return ptr + 1
end

function _addcol!(
        rowval::Vector{Int},
        nzval::Vector{T},
        ind::Vector{Int},
        A::Matrix{T},
        loccol::Int,
        noffrow::Int,
        ptr::Int,
    ) where {T}
    for locrow in axes(A, 1)
        rowval[ptr] = noffrow + locrow
        nzval[ptr] = A[locrow, loccol]
        ind[(loccol - 1) * size(A, 1) + locrow] = ptr
        ptr += 1
    end

    return ptr
end

function _addcol!(
    rowval::Vector{Int},
    nzval::Vector{T},
    ind::Vector{Int},
    A::UniformBlockDiagonal{T},
    loccol::Int,
    noffrow::Int,
    ptr::Int,
) where {T}
    nsubrow, nsubcol, nlevel = size(A.data)
    level = (loccol - 1) ÷ nsubcol + 1
    subcol = (loccol - 1) % nsubcol + 1
    noffrow += (level - 1) * nsubrow

    for locrow in 1:nsubrow
        rowval[ptr] = noffrow + locrow
        nzval[ptr] = A.data[locrow, subcol, level]
        ind[(level - 1) * nsubrow * nsubcol + (subcol - 1) * nsubrow + locrow] = ptr
        ptr += 1
    end

    return ptr
end

function _addcol!(
    rowval::Vector{Int},
    nzval::Vector{T},
    ind::Vector{Int},
    A::BlockedSparse{T},
    loccol::Int,
    noffrow::Int,
    ptr::Int,
) where {T}
    return _addcol!(rowval, nzval, ind, A.cscmat, loccol, noffrow, ptr)
end

function _addcol!(
    rowval::Vector{Int},
    nzval::Vector{T},
    ind::Vector{Int},
    A::SparseMatrixCSC{T},
    loccol::Int,
    noffrow::Int,
    ptr::Int,
) where {T}
    for p in nzrange(A, loccol)
        locrow = rowvals(A)[p]
        rowval[ptr] = noffrow + locrow
        nzval[ptr] = nonzeros(A)[p]
        ind[p] = ptr; ptr += 1
    end

    return ptr
end

function _addblk!(nzval::Vector{T}, A::AbstractMatrix{T}, indices::Vector{Int}) where {T}
    src = _nonzeros(A)

    for k in eachindex(indices)
        nzval[indices[k]] = src[k]
    end

    return
end

function _init(M::LinearMixedModel{T}; kw...) where {T}
    # get sparsity pattern of `M`
    S, indices = _csc(M.A, M.reterms)

    # construct clique tree
    perm, tree = cliquetree(Symmetric(S, :L); kw...)

    # add fixed-effects columns to `clique`
    nfixcol = size(M.A[end], 1)
    clique = Vector{Int}(undef, nfixcol)

    j = 0

    for (i, v) in enumerate(perm)
        if nfixcol + v > size(S, 1)
            j += 1; clique[j] = i
        end
    end

    # rotate clique tree so that fixed-effects columns
    # are eliminated last
    permute!(perm, cliquetree!(tree, clique))

    # construct uninitialized Cholesky factor
    F = ChordalCholesky{:L, T}(perm, ChordalSymbolic(tree))

    # construct workkspace
    W = ChordalWorkspace{T}(map(_similar, M.A), S, indices)
    return F, W
end

function _logdet(M::LinearMixedModel, F::ChordalCholesky)
    n = size(F.L, 1) - size(M.A[end], 1)
    D = view(diag(F.L), 1:n)
    return 2sum(log ∘ abs, D)
end

function _transform!(
        M::LinearMixedModel{T},
        F::ChordalCholesky{UPLO, T},
        W::ChordalWorkspace{T},
    ) where {UPLO, T}
    # apply Λ-transform to M, writing the result to `W.blocks`.
    for k in eachindex(M.reterms)
        R = M.reterms[k]
        blkptr = kp1choose2(k)
        copyscaleinflate!(W.blocks[blkptr], M.A[blkptr], R)

        for i in k + 1:length(M.reterms) + 1
            blkptr = block(i, k)
            rmulΛ!(copyto!(W.blocks[blkptr], M.A[blkptr]), R)
        end

        for j in 1:k - 1
            blkptr = block(k, j)
            lmulΛ!(R', W.blocks[blkptr])
        end
    end

    blkptr = kp1choose2(length(M.reterms) + 1)
    copyto!(W.blocks[blkptr], M.A[blkptr])

    # write `W.blocks` to `W.matrix`
    for blkptr in eachindex(W.blocks, W.indices)
        _addblk!(nonzeros(W.matrix), W.blocks[blkptr], W.indices[blkptr])
    end

    # write `W.matrix` to `F`.
    copy!(F, sparse(Symmetric(W.matrix, :L)))
    return F
end
