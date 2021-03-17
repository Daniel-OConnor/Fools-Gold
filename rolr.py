import torch

def ROLR(simulator, r_hat, prior, N, epochs, lr):
    optimizer = torch.optim.Adam((r_hat.parameters()), lr=lr)
    mse = torch.nn.MSELoss()
    loss_average = 0
    for i in range(epochs):
        x0 = []
        x1 = []
        r_score0 = []
        r_score1 = []
        theta0 = []
        theta1 = []
        for _ in range(N):
            t0 = prior()
            t1 = prior()
            theta0.append(t0)
            theta1.append(t1)

            sample0 = simulator.sample_R(t1, t0)
            x0.append(sample0[0])
            r_score0.append(sample0[1])

            sample1 = simulator.sample_R(t0, t1)
            x1.append(sample1[0])
            r_score1.append(sample1[1])

        with torch.no_grad():
            x0 = torch.stack(x0)
            x1 = torch.stack(x1)
            r_score0 = torch.stack(r_score0)
            r_score1 = torch.stack(r_score1)
            theta0 = torch.stack(theta0)
            theta1 = torch.stack(theta1)
        # Not sure if this is the correct formula for loss0
        loss0 = mse(r_hat(x0, theta1, theta0), r_score0)
        loss1 = mse(r_hat(x1, theta0, theta1), r_score1)
        loss = (loss0 + loss1) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_average += loss.detach().cpu().numpy()
        if i < 5:
            print(loss.detach().cpu().numpy())
        if i % 250 == 249:
            print(loss_average / 250)
            loss_average = 0
