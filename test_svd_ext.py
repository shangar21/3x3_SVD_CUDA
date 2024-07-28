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
    u, s, v = svd_extension.svd_cuda(input_flat)
    
    # Reshape S into a diagonal matrix without explicit iteration
    #s_matrix = torch.zeros((N, 3, 3), dtype=torch.float32).cuda()
    #s_matrix[:, range(3), range(3)] = s
    
    return u.view(N, 3, 3), s, v.view(N, 3, 3)

def print_matrix(matrix, name):
    print(f"Matrix {name}:")
    print(matrix)
    print()

def test_svd(U, S, V, original_matrices):
    """
    Tests the correctness of the SVD decomposition.

    Args:
        U (torch.Tensor): A tensor of shape (N, 3, 3) containing the left singular vectors.
        S (torch.Tensor): A tensor of shape (N, 3, 3) containing the diagonal matrices of singular values.
        V (torch.Tensor): A tensor of shape (N, 3, 3) containing the right singular vectors.
        original_matrices (torch.Tensor): A tensor of shape (N, 3, 3) containing the original matrices.

    Returns:
        tuple: Two float values representing the sum of the differences for the reconstruction and orthogonality checks.
    """
    N = original_matrices.size(0)
    
    # Convert S from (N, 3) to (N, 3, 3) diagonal matrices
    S_matrix = torch.zeros((N, 3, 3), dtype=torch.float32).cuda()
    S_matrix[:, range(3), range(3)] = S

    # Verify U * S * V^T reconstructs the original matrix
    reconstructed_matrices = torch.bmm(U, torch.bmm(S_matrix, V.transpose(1, 2)))

    print(reconstructed_matrices)
    print(original_matrices)
    
    # Compute the sum of differences for the reconstruction check
    reconstruction_diff = torch.sum(torch.abs(reconstructed_matrices - original_matrices))
    
    # Check orthogonality of U and V
    identity_u = torch.bmm(U, U.transpose(1, 2))
    identity_v = torch.bmm(V, V.transpose(1, 2))
    
    identity_matrix = torch.eye(3, dtype=torch.float32).cuda().unsqueeze(0).repeat(N, 1, 1)
    
    # Compute the sum of differences for the orthogonality check
    orthogonality_diff_u = torch.sum(torch.abs(identity_u - identity_matrix))
    orthogonality_diff_v = torch.sum(torch.abs(identity_v - identity_matrix))
    
    total_orthogonality_diff = orthogonality_diff_u + orthogonality_diff_v
    return reconstruction_diff.item(), total_orthogonality_diff.item(), reconstructed_matrices 

def find_max_error_matrix(original_matrices, reconstructed_matrices):
    """
    Finds the matrix with the largest error between the original and reconstructed matrices.
    
    Args:
        original_matrices (torch.Tensor): A tensor of shape (N, 3, 3) containing the original matrices.
        reconstructed_matrices (torch.Tensor): A tensor of shape (N, 3, 3) containing the reconstructed matrices.
        
    Returns:
        tuple: The index of the matrix with the largest error and the maximum error value.
    """
    # Calculate the error for each matrix
    errors = torch.norm(original_matrices - reconstructed_matrices, dim=(1, 2))
    
    # Find the index of the matrix with the largest error
    max_error_index = torch.argmax(errors).item()
    max_error_value = errors[max_error_index].item()
    
    return max_error_index, max_error_value

def main():
    # Example batch size
    N = 20_000
    # Define a batch of input matrices (N x 3 x 3)
    original_matrices = torch.tensor([
        [
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0]
        ],
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ],
        
        [
            [1.1, 2.1, 3.1],
            [4.2, 5.2, 6.2],
            [7.3, 8.3, 9.3]
        ]
    ], dtype=torch.float32).cuda()

    original_matrices = torch.randn(N, 3, 3).cuda()


    # Perform batch SVD
    U, S, V = batch_svd(original_matrices)
    torch.cuda.synchronize()

    # Test the SVD results
    reconstruction_diff, orthogonality_diff, reconstructed_matrices = test_svd(U, S, V, original_matrices)
    max_error_index, max_error_value = find_max_error_matrix(original_matrices, reconstructed_matrices)
    print("MATRICES WITH MAX ERROR")
    print(original_matrices[max_error_index])
    print(reconstructed_matrices[max_error_index])
    print(f"Reconstruction difference sum: {reconstruction_diff}")
    print(f"Orthogonality difference sum: {orthogonality_diff}")


if __name__ == "__main__":
    main()

