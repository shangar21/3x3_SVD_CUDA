import torch
import svd_extension

def batch_svd(input_matrices):
    """
    Performs SVD on a batch of 3x3 matrices.

    Args:
        input_matrices (torch.Tensor): A tensor of shape (N, 3, 3) containing the batch of 3x3 matrices.

    Returns:
        tuple: Three tensors U, S, V where:
               - U is a tensor of shape (N, 3, 3) containing the left singular vectors.
               - S is a tensor of shape (N, 3, 3) containing the diagonal matrices of singular values.
               - V is a tensor of shape (N, 3, 3) containing the right singular vectors.
    """
    N = input_matrices.size(0)
    input_flat = input_matrices.view(N, -1)
    
    # Perform SVD using the custom extension
    u, s, v = svd_extension.svd_cuda_batch(input_flat)
    
    # Reshape S into a diagonal matrix without explicit iteration
    s_matrix = torch.zeros((N, 3, 3), dtype=torch.float32).cuda()
    s_matrix[:, range(3), range(3)] = s
    
    return u.view(N, 3, 3), s_matrix, v.view(N, 3, 3)

