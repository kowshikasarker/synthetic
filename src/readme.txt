#gcvae

hidden_dim = [4, 6, 8]
lr = [1e-4, 1e-5, 1e-6]
batch_size = [32, 64]

total_steps=10 -> call annealer.step after every epoch, rather than after every batch
only sp edges

#cvae

hidden_dim = [|f|//2, |f|//3, |f|//4]
lr = [1e-4, 1e-5, 1e-6]
batch_size = [32, 64]

total_steps=10

