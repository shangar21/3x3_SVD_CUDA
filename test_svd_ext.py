import torch
import svd_extension

def print_matrix(matrix, name):
    print(f"Matrix {name}:")
    print(matrix)
    print()

# Define the input matrix (3x3)
input_matrix = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
], dtype=torch.float32).cuda()

# Flatten the input matrix to pass to the kernel
input_flat = input_matrix.flatten()

# Perform SVD using the custom extension
u, s, v = svd_extension.svd_cuda(input_flat)

# Reshape S into a diagonal matrix
s_matrix = torch.diag(s)

# Reshape U and V to 3x3 matrices
u_matrix = u.view(3, 3)
v_matrix = v.view(3, 3)

# Print the results
print_matrix(u_matrix, "U")
print_matrix(s_matrix, "S")
print_matrix(v_matrix, "V")

# Verify U * S * V^T reconstructs the original matrix
reconstructed_matrix = torch.mm(torch.mm(u_matrix, s_matrix), v_matrix.t())
print_matrix(reconstructed_matrix, "Reconstructed")

# Check orthogonality of U and V
identity_u = torch.mm(u_matrix, u_matrix.t())
identity_v = torch.mm(v_matrix, v_matrix.t())

# Print the identity matrices
print_matrix(identity_u, "U * U^T")
print_matrix(identity_v, "V * V^T")

# Verify if the reconstructed matrix is close to the original matrix
if torch.allclose(reconstructed_matrix, input_matrix, atol=1e-6):
    print("Reconstructed matrix is close to the original matrix.")
else:
    print("Reconstructed matrix is not close to the original matrix.")

# Verify if U and V are orthogonal
identity_matrix = torch.eye(3, dtype=torch.float32).cuda()
if torch.allclose(identity_u, identity_matrix, atol=1e-6):
    print("U is orthogonal.")
else:
    print("U is not orthogonal.")

if torch.allclose(identity_v, identity_matrix, atol=1e-6):
    print("V is orthogonal.")
else:
    print("V is not orthogonal.")

