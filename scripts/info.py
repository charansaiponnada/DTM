import laspy
import numpy as np

"""
Requirements:
laspy[lazrs]
laspy
pyproj
"""
def analyze_las(file_path):
    print("="*60)
    print(f"File: {file_path}\n")

    las = laspy.read(file_path)

    #  Dimensions
    dims = list(las.point_format.dimension_names)
    print("Dimensions:")
    print(dims)

    #  CRS
    print("\nCRS:")
    print(las.header.parse_crs())

    #  Number of points
    n_points = len(las.points)
    print("\nNumber of points:", n_points)

    #  Bounding box
    min_x, max_x = las.x.min(), las.x.max()
    min_y, max_y = las.y.min(), las.y.max()
    area = (max_x - min_x) * (max_y - min_y)

    print("\nBounding Box:")
    print(f"X: {min_x:.2f} → {max_x:.2f}")
    print(f"Y: {min_y:.2f} → {max_y:.2f}")
    print(f"Approx Area: {area:.2f} sq.units")

    if area > 0:
        print(f"Estimated Density: {n_points/area:.2f} points per sq.unit")

    # Classification check
    if "classification" in dims:
        unique_classes = np.unique(las.classification)
        print("\nClassification exists.")
        print("Classes found:", unique_classes)

        if 2 in unique_classes:
            print("✔ Ground class (2) present")
        else:
            print("✘ Ground class (2) NOT found")
    else:
        print("\n✘ No classification field found")

    # Intensity check
    if "intensity" in dims:
        min_int = las.intensity.min()
        max_int = las.intensity.max()
        print("\nIntensity exists.")
        print(f"Range: {min_int} → {max_int}")

        if max_int == 0:
            print("⚠ Intensity values are all zero")
    else:
        print("\n✘ No intensity field found")

    print("="*60, "\n")


# Run on both files
# change the path if needed.
analyze_las("KHAPRETA_510206.laz")
analyze_las("DEVDI_POINT CLOUD (511671).las")