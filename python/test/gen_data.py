from test import *


def gen_data(n, d, k, m, D):

    X = create_mock_mnist_dataset(n=n, d=d)
    A, B = create_mock_A_B_matrices(n=n, k=k, m=m, D=D)

    cpArrayToFlatCSV(X, f"../data/X_n{n}_d{d}.csv")
    cpArrayToFlatCSV(A, f"../data/A_n{n}_k{k}.csv")
    cpArrayToFlatCSV(B, f"../data/B_D{D}_m{m}.csv")

    distances = gsd.find_distances(X, A, B)
    cp.cuda.Device().synchronize()
    cpArrayToFlatCSV(distances, f"../data/distances_n{n}_k{k}_m{m}.csv")

n = 1_000
d = 20
k = 3
m = 20
D = 100
gen_data(n, d, k, m, D)