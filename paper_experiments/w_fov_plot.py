import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

kf_list = [10, 20]
alpha_deg_list = [90]

plt.figure(figsize=(12,3))

# 左图
plt.subplot(1,2,1)
for alpha_deg in alpha_deg_list:
    alpha = np.deg2rad(alpha_deg)

    theta = np.linspace(-np.deg2rad(180), np.deg2rad(180), 500)
    cos_theta = np.cos(theta)
    for k_f in kf_list:
        w_fov = 1 / (1 + np.exp(-k_f * (cos_theta - np.cos(alpha))))
        plt.plot(np.rad2deg(theta), w_fov, label=f'kf={k_f}, alpha={alpha_deg}')

plt.xlabel(r'$\theta_j$ (degrees)')
plt.ylabel(r'$w_{\mathrm{fov}}(\theta_j)$')
plt.title('FOV Weight vs Theta')
plt.grid(True)
plt.legend()
plt.xlim(-180, 180)

# 右图
plt.subplot(1,2,2)
for alpha_deg in alpha_deg_list:
    alpha = np.deg2rad(alpha_deg)
    theta = np.linspace(-np.deg2rad(180), np.deg2rad(180), 500)
    cos_theta = np.cos(theta)
    for k_f in kf_list:
        w_fov = 1 / (1 + np.exp(-k_f * (cos_theta - np.cos(alpha))))
        w_fov_prime = k_f * w_fov * (1 - w_fov)
        plt.plot(cos_theta, w_fov_prime, label=f'kf={k_f}, alpha={alpha_deg}')

plt.xlabel(r'$\cos\theta_j$')
plt.ylabel(r'$\partial w_{\mathrm{fov}} / \partial \cos\theta_j$')
plt.title('Derivative of FOV Weight')
plt.grid(True)
plt.legend()
plt.xlim(-1, 1)

plt.tight_layout()

# 保存
plt.savefig("fov_weight_analysis.pdf", bbox_inches="tight")
plt.show()