import numpy as np
import sys

# Import your class from forward_kinematics.py
try:
    from forward_kinematics import ForwardKinematics
except ImportError:
    print("Error: Could not import ForwardKinematics from forward_kinematics.py.")
    print("Make sure this test script is in the same directory as forward_kinematics.py.")
    sys.exit(1)


def run_tests():
    # Instantiate without calling rclpy by using __new__ to avoid ROS node initialization
    fk = ForwardKinematics.__new__(ForwardKinematics)

    passed_all = True

    def test(name, condition, details=""):
        nonlocal passed_all
        if condition:
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name}")
            if details:
                print(f"       Details: {details}")
            passed_all = False

    print("\n--- Testing translation(x, y, z) ---")
    T = fk.translation(1.0, 2.0, 3.0)
    test("Shape is (4, 4)", T.shape == (4, 4))
    
    # Check bottom row is [0, 0, 0, 1]
    test("Bottom row is [0, 0, 0, 1]", np.allclose(T[3, :], [0, 0, 0, 1]))
    
    # Translating the origin [0, 0, 0, 1]^T should result in [1, 2, 3, 1]^T
    origin = np.array([0.0, 0.0, 0.0, 1.0])
    transformed_origin = T @ origin
    test(
        "Translates origin to (1, 2, 3)",
        np.allclose(transformed_origin, [1.0, 2.0, 3.0, 1.0]),
        f"Got: {transformed_origin[:3]}",
    )

    print("\n--- Testing rotation_y(angle) ---")
    R_y_0 = fk.rotation_y(0.0)
    test("rotation_y(0) is 4x4 Identity", np.allclose(R_y_0, np.eye(4)))

    # Rotating +90 degrees (pi / 2 rad) around Y:
    # Right-hand rule around +Y:
    # +Z axis rotates into +X axis: [0, 0, 1, 1] -> [1, 0, 0, 1]
    # +X axis rotates into -Z axis: [1, 0, 0, 1] -> [0, 0, -1, 1]
    # Y axis remains unchanged:     [0, 1, 0, 1] -> [0, 1, 0, 1]
    pi_2 = np.pi / 2.0
    R_y_90 = fk.rotation_y(pi_2)

    vec_z = np.array([0.0, 0.0, 1.0, 1.0])
    res_z = R_y_90 @ vec_z
    test(
        "rotation_y(+90 deg) maps +Z to +X",
        np.allclose(res_z, [1.0, 0.0, 0.0, 1.0]),
        f"Expected [1, 0, 0, 1], got: {np.round(res_z, 3)}",
    )

    vec_x = np.array([1.0, 0.0, 0.0, 1.0])
    res_x = R_y_90 @ vec_x
    test(
        "rotation_y(+90 deg) maps +X to -Z",
        np.allclose(res_x, [0.0, 0.0, -1.0, 1.0]),
        f"Expected [0, 0, -1, 1], got: {np.round(res_x, 3)}",
    )

    # Check 3x3 rotation block is orthogonal: R @ R.T == I
    test(
        "rotation_y 3x3 block is orthogonal",
        np.allclose(R_y_90[:3, :3] @ R_y_90[:3, :3].T, np.eye(3)),
    )

    print("\n--- Testing rotation_z(angle) ---")
    R_z_0 = fk.rotation_z(0.0)
    test("rotation_z(0) is 4x4 Identity", np.allclose(R_z_0, np.eye(4)))

    # Rotating +90 degrees around Z:
    # +X axis rotates into +Y axis: [1, 0, 0, 1] -> [0, 1, 0, 1]
    # +Y axis rotates into -X axis: [0, 1, 0, 1] -> [-1, 0, 0, 1]
    # Z axis remains unchanged:     [0, 0, 1, 1] -> [0, 0, 1, 1]
    R_z_90 = fk.rotation_z(pi_2)

    res_zx = R_z_90 @ vec_x
    test(
        "rotation_z(+90 deg) maps +X to +Y",
        np.allclose(res_zx, [0.0, 1.0, 0.0, 1.0]),
        f"Expected [0, 1, 0, 1], got: {np.round(res_zx, 3)}",
    )

    vec_y = np.array([0.0, 1.0, 0.0, 1.0])
    res_zy = R_z_90 @ vec_y
    test(
        "rotation_z(+90 deg) maps +Y to -X",
        np.allclose(res_zy, [-1.0, 0.0, 0.0, 1.0]),
        f"Expected [-1, 0, 0, 1], got: {np.round(res_zy, 3)}",
    )

    test(
        "rotation_z 3x3 block is orthogonal",
        np.allclose(R_z_90[:3, :3] @ R_z_90[:3, :3].T, np.eye(3)),
    )

    print("\n" + "=" * 40)
    if passed_all:
        print("ALL TESTS PASSED! Transformations are correct.")
    else:
        print("SOME TESTS FAILED. Check the signs in your matrix rows/cols.")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_tests()