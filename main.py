import numpy as np
import numpy.linalg as linalg
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import matplotlib.cm as cm
import scipy.stats as sc

colors = cm.rainbow(np.linspace(0, 1, 10))
X = np.loadtxt('../../Downloads/datax.txt')
Y = np.loadtxt('../../Downloads/datay.txt')
Y = Y - np.mean(Y)


def transpose_square(X):
    return np.dot(X.transpose(), X)


def mediannorm(A):
    return np.median(A, 1)


def normal_with_max(A):
    A = np.array(A)
    return np.sum(np.abs(A), 1) / np.max(np.sum(np.abs(A), 1))


def plot_subgraph_fig1(x_axis, y_axis, color):
    xy = [x_axis, y_axis]
    xy = np.array(xy)
    xy = xy[:, xy[0, :].argsort()]
    plt.plot(xy[0], xy[1], c=color)


def Select_lam(n, r, delta, lam, X, Y):
    # Initialization
    Y = Y - np.mean(Y)
    beta = [np.random.uniform(size=X.shape[1])]
    sigma_sq = [np.random.uniform()]
    tau_sq = [np.random.uniform(size=X.shape[1])]
    lambda_sq = [lam ** 2] * n
    for i in range(n):
        D_tau = np.diag(tau_sq[i])
        A = X.transpose().dot(X) + np.linalg.inv(D_tau)
        multi_norm_mean = np.linalg.inv(A).dot(X.transpose()).dot(Y)
        multi_norm_cov = sigma_sq[i] * np.linalg.inv(A)
        beta.append(np.random.multivariate_normal(multi_norm_mean, multi_norm_cov))
        shape = (X.shape[0] - 1 + X.shape[1]) / 2
        scale = ((Y - X.dot(beta[i + 1])).dot((Y - X.dot(beta[i + 1]))) + beta[i + 1].transpose().dot(
            np.linalg.inv(D_tau)).dot(beta[i + 1])) / 2
        sigma_sq.append(sc.invgamma.rvs(a=shape, scale=scale))
        mean = np.sqrt(lambda_sq[i] * sigma_sq[i + 1] / beta[i + 1] ** 2)
        scale = np.repeat(lambda_sq[i], X.shape[1])
        tau_sq.append(1 / np.random.wald(mean, scale))
    return beta


def Gibbs_sampler_lambda(n, r, delta, lam, x, y):
    X = x  # np.loadtxt('../../Downloads/datax.txt')
    Y = y  # np.loadtxt('../../Downloads/datay.txt')
    beta = [np.random.uniform(size=X.shape[1])]
    sigma_sq = [np.random.uniform()]
    tau_sq = [np.random.uniform(size=X.shape[1])]
    lambda_sq = [lam]
    for i in range(n):
        D_tau = np.diag(tau_sq[i])
        A = X.transpose().dot(X) + np.linalg.inv(D_tau)
        multi_norm_mean = np.linalg.inv(A).dot(X.transpose()).dot(Y)
        multi_norm_cov = sigma_sq[i] * np.linalg.inv(A)
        beta.append(np.random.multivariate_normal(multi_norm_mean, multi_norm_cov))
        shape = (X.shape[0] - 1 + X.shape[1]) / 2
        scale = ((Y - X.dot(beta[i + 1])).dot((Y - X.dot(beta[i + 1]))) + beta[i + 1].transpose().dot(
            np.linalg.inv(D_tau)).dot(beta[i + 1])) / 2
        sigma_sq.append(sc.invgamma.rvs(a=shape, scale=scale))
        mean = np.sqrt(lambda_sq[i] * sigma_sq[i + 1] / beta[i + 1] ** 2)
        scale = np.repeat(lambda_sq[i], X.shape[1])
        tau_sq.append(1 / np.random.wald(mean, scale))
        shape = X.shape[1] + r
        rate = sum(tau_sq[i + 1]) / 2 + delta
        lambda_sq.append(np.random.gamma(shape, 1 / rate))
    return beta, lambda_sq


def Optimize(type, choose_lam):
    ##initialize
    Beta = []
    alphas = np.power(1.6, np.arange(-40, 0)) * 45
    score = []
    if choose_lam == True:
        lam = 0.237
        if type == 'ols':
            clf = linear_model.LinearRegression()
            clf.fit(X, Y)
            beta = clf.coef_
            return beta

        elif type == 'bayes':
            Beta = Gibbs_sampler_lambda(10000, 1, 1.78, lam, X, Y)
            Beta = np.array(Beta)
            return Beta

        elif type == 'lasso':
            for alpha in alphas:
                clf = linear_model.Lasso(alpha=alpha)
                clf.fit(X, Y)
                beta = clf.coef_
                Beta.append(beta)
                score.append(np.mean(cross_val_score(clf, X, Y, cv=50)))
            Beta = np.array(Beta)
            return Beta, score


    else:
        if type == 'lasso':
            for alpha in alphas:
                clf = linear_model.Lasso(alpha=alpha)
                clf.fit(X, Y)
                beta = clf.coef_
                Beta.append(beta)
                # score = np.mean(cross_val_score(clf, X, Y, cv=5))
            Beta = np.array(Beta)
            return Beta

        elif type == 'rigid':
            for alpha in alphas:
                clf = linear_model.Ridge(alpha=alpha)
                clf.fit(X, Y)
                beta = clf.coef_
                Beta.append(beta)
            Beta = np.array(Beta)

            return Beta

        elif type == 'bayes':
            for lam in alphas:
                print(lam)
                Beta.append(Gibbs_sampler_lambda(10000, 1, 1.78,  lam, X, Y))
            Beta = np.array(Beta)
            return Beta


if __name__ == "__main__":
    ### plot figure 4
    # lasso
    l_beta = Optimize('lasso', False)
    x_axis = normal_with_max(l_beta)
    for i in range(10):
        y_axis = l_beta[:, i]
        plot_subgraph_fig1(x_axis, y_axis, colors[i])
    plt.show()

    # rigid
    r_beta = Optimize('rigid', False)
    x_axis = normal_with_max(r_beta)
    for i in range(10):
        y_axis = r_beta[:, i]
        plot_subgraph_fig1(x_axis, y_axis, colors[i])
    plt.show()

    # bayes
    B_beta = Optimize('bayes', False)
    print(B_beta.shape)
    B_beta = np.array(B_beta)
    x_axis = normal_with_max(B_beta)
    for i in range(10):
        y_axis = B_beta[:, i]
        plot_subgraph_fig1(x_axis, y_axis, colors[i])
    plt.show()

    ###plot figure 5
    # lam = Select_lam(10000, 1, 1.78)
    # print(np.sqrt(lam))  ####0.237

    # ols
    o_beta = Optimize('ols', True)
    plt.scatter(o_beta, np.arange(10), marker='^', label='ols')

    # bayes lasso
    B_beta = Optimize('bayes', True)
    plt.scatter(np.median(B_beta[-1000:, ], 0), np.arange(10), marker='*', label='bayes')
    for i in range(10):
        matrix = np.sort(B_beta[:, i])
        plt.hlines(i, matrix[int(0.025 * 10000)], matrix[int(0.975 * 10000)])

    # nfolder lasso
    l_beta, score = Optimize('lasso', True)
    l_beta_new = l_beta[score.index(max(score)), :]
    plt.scatter(l_beta_new, np.arange(10) + 0.2, marker='s', label='n_folder-lasso')

    # nfolder_lasso
    l1norm_Bbeta = np.sum(np.abs(np.median(B_beta[-1000:, ], 0)))
    l1norm_dif = np.abs(np.sum(np.abs(l_beta), 1) - l1norm_Bbeta)
    best_matchl1norm_l_beta = l_beta[np.where(l1norm_dif == np.min(l1norm_dif))[0][0], :]
    plt.scatter(best_matchl1norm_l_beta, np.arange(10) - 0.2, marker='h', label='l1_best_match')
    plt.legend()
    plt.show()

    ####simulation result
    x = np.random.uniform(size=(1100, 5))
    y = np.random.uniform(size=1100)
    train_MSE_bayes = []
    train_MSE_ols = []
    test_MSE_bayes = []
    test_MSE_ols = []
    for i in range(10):
        b_beta, lam = Select_lam(100, 1, 1, x[100 * i:100 * i + 100, :], y[100 * i:100 * i + 100])
        b_beta = np.array(b_beta)
        b_beta = b_beta[-1, :]
        clf = linear_model.LinearRegression()
        clf.fit(x[100 * i:100 * i + 100, :], y[100 * i:100 * i + 100])
        train_MSE_bayes.append(
            np.linalg.norm(y[100 * i:100 * i + 100] - clf.predict(x[100 * i:100 * i + 100, :])) ** 2)
        train_MSE_ols.append(
            np.linalg.norm(y[100 * i:100 * i + 100] - np.dot(x[100 * i:100 * i + 100, :], b_beta)) ** 2)
        test_MSE_bayes.append(np.linalg.norm(y[1000:1100] - clf.predict(x[1000:1100, :])) ** 2)
        test_MSE_ols.append(np.linalg.norm(y[1000:1100] - np.dot(x[1000:1100, :], b_beta)) ** 2)
    plt.plot(np.arange(10), train_MSE_bayes, c='r', label='bayes_train')
    plt.plot(np.arange(10), train_MSE_ols, c='b', label='ols_train')
    plt.legend()
    plt.show()
    plt.close()
    plt.plot(np.arange(10), test_MSE_bayes, c='r', label='bayes_test')
    plt.plot(np.arange(10), test_MSE_ols, c='b', label='ols_test')
    plt.legend()
    plt.show()

    ####choose lambda
    Y = Y - np.mean(Y)
    k = 0
    for init_lam in [0.01, 1, 10, 100]:
        lam = []
        for i in range(30):
            if i == 0:
                lambda_ = np.sqrt(
                    2 * X.shape[1] / sum(np.mean(Gibbs_sampler_lambda(1000, 1, 1.78, init_lam)[1], axis=0)))
                lam.append(lambda_)
            else:
                lambda_ = np.sqrt(
                    2 * X.shape[1] / sum(np.mean(Gibbs_sampler_lambda(1000, 1, 1.78, lambda_)[1], axis=0)))
                lam.append(lambda_)
            print(i)

        plt.plot(np.arange(30), lam, c=colors[k], label='init = %f' % init_lam)
        k = k + 1
    plt.legend()
    plt.show()
