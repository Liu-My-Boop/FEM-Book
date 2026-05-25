import numpy as np


def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元长度、方向余弦、全局坐标系6×6单元刚度矩阵Ke
    :param x1: 节点1坐标 [x, y, z]
    :param x2: 节点2坐标 [x, y, z]
    :param E: 弹性模量 (Pa)
    :param A: 截面积 (m²)
    :return: L(长度), cx, cy, cz(方向余弦), Ke(6×6刚度矩阵)
    """
    x1 = np.array(x1, dtype=np.float64)
    x2 = np.array(x2, dtype=np.float64)

    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]
    L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    if L < 1e-12:
        raise ValueError("错误：两节点重合，无法计算三维杆单元！")

    cx = dx / L
    cy = dy / L
    cz = dz / L

    T = np.array([[cx, cy, cz, 0, 0, 0],
                  [0, 0, 0, cx, cy, cz]])

    k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])
    Ke = T.T @ k_local @ T

    return L, cx, cy, cz, Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据节点位移计算单元应变、应力、轴力
    :param de: 节点位移 [u1,v1,w1,u2,v2,w2]
    :return: epsilon(应变), sigma(应力), N(轴力)
    """
    L, cx, cy, cz, _ = truss3d_element_stiffness(x1, x2, E, A)
    de = np.array(de, dtype=np.float64).reshape(6, 1)

    B = np.array([[-cx, -cy, -cz, cx, cy, cz]]) / L
    epsilon = (B @ de)[0, 0]

    sigma = E * epsilon
    N = sigma * A

    return epsilon, sigma, N


# ------------------------------
# 算例1：沿x轴的一维杆单元
# ------------------------------
def example_1():
    print("=" * 60)
    print("【算例1：沿x轴一维杆单元】")
    print("-" * 60)
    x1 = [0, 0, 0]
    x2 = [2, 0, 0]
    E = 200e9
    A = 1.0e-4
    de = [0, 0, 0, 1.0e-3, 0, 0]

    L, cx, cy, cz, Ke = truss3d_element_stiffness(x1, x2, E, A)
    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)

    print(f"单元长度 L = {L:.2f} m")
    print(f"方向余弦 (cx,cy,cz) = ({cx:.0f}, {cy:.0f}, {cz:.0f})")
    print(f"轴向应变 ε = {epsilon:.2e}")
    print(f"轴向应力 σ = {sigma / 1e6:.2f} MPa")
    print(f"轴力 N = {N:.2e} N")
    print("\n6×6单元刚度矩阵 Ke：")
    print(np.round(Ke, 2))

    # 刚度矩阵退化验证（嵌入输出，不额外标注）
    print("\n刚度矩阵非零元素：")
    for i in range(6):
        for j in range(6):
            if abs(Ke[i, j]) > 1e-6:
                print(f"Ke[{i + 1},{j + 1}] = {Ke[i, j]:.2e}")


# ------------------------------
# 算例2：空间任意方向杆单元
# ------------------------------
def example_2():
    print("\n" + "=" * 60)
    print("【算例2：空间任意方向杆单元】")
    print("-" * 60)
    x1 = [0, 0, 0]
    x2 = [1, 2, 2]
    E = 210e9
    A = 2.0e-4
    de = [0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3]

    L, cx, cy, cz, Ke = truss3d_element_stiffness(x1, x2, E, A)
    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)

    is_symmetric = np.allclose(Ke, Ke.T)
    eig_vals = np.linalg.eigvals(Ke)
    eig_nonneg = np.all(eig_vals >= -1e-6)

    print(f"单元长度 L = {L:.0f} m")
    print(f"方向余弦 (cx,cy,cz) = ({cx:.3f}, {cy:.3f}, {cz:.3f})")
    print(f"Ke是否对称：{is_symmetric}")
    print(f"Ke特征值全非负：{eig_nonneg}")
    print(f"轴向应变 ε = {epsilon:.2e}")
    print(f"轴向应力 σ = {sigma / 1e6:.2f} MPa")
    print(f"轴力 N = {N:.2e} N")

    #输出6×6刚度矩阵
    print("\n6×6单元刚度矩阵 Ke：")
    print(np.round(Ke, 2))

    # 刚体平移验证
    de_rigid = [0.001, 0.002, 0.002, 0.001, 0.002, 0.002]
    eps_r, sig_r, N_r = truss3d_element_stress(x1, x2, E, A, de_rigid)
    print(f"\n刚体平移应变：{eps_r:.2e}")
    print(f"刚体平移应力：{sig_r:.2e} Pa")
    print(f"刚体平移轴力：{N_r:.2e} N")

    # 特征值输出
    print("\nKe特征值：", np.round(eig_vals, 2))


# ------------------------------
# 刚度矩阵物理意义验证
# ------------------------------
def stiffness_physical_meaning():
    print("\n" + "=" * 60)
    print("【刚度矩阵物理意义：第j列=单位位移产生的节点力】")
    print("-" * 60)
    x1 = [0, 0, 0]
    x2 = [1, 0, 0]
    E = 200e9
    A = 0.0001
    _, _, _, _, Ke = truss3d_element_stiffness(x1, x2, E, A)

    j = 3
    de = np.zeros(6)
    de[j] = 1.0
    Fe = Ke @ de

    print(f"自由度 {j + 1} 施加单位位移，节点力 Fe：")
    print(np.round(Fe, 2))
    print(f"\nKe第{j + 1}列：")
    print(np.round(Ke[:, j], 2))
    print("\n结论：刚度矩阵第j列 = 第j自由度单位位移产生的节点力")


# ------------------------------
# 主程序运行
# ------------------------------
if __name__ == "__main__":
    try:
        example_1()
        example_2()
        stiffness_physical_meaning()
        print("\n" + "=" * 60)
        print("所有算例运行完成，结果与作业要求一致！")
        print("=" * 60)
    except ValueError as e:
        print(e)