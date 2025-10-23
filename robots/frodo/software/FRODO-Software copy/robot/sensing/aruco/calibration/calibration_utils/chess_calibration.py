# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Perform camera calibration using a chessboard.

import cv2  # Import the OpenCV library to enable computer vision
import numpy as np  # Import the NumPy scientific computing library
import glob  # Used to get retrieve files that have a specified pattern
import yaml

from utils.files import splitExtension

# Chessboard dimensions
number_of_squares_X = 10  # Number of chessboard squares along the x-axis
number_of_squares_Y = 7  # Number of chessboard squares along the y-axis
nX = number_of_squares_X - 1  # Number of interior corners along x-axis
nY = number_of_squares_Y - 1  # Number of interior corners along y-axis
square_size = 0.025  # Length of the side of a square in meters

# Store vectors of 3D points for all chessboard images (world coordinate frame)
object_points = []

# Store vectors of 2D points for all chessboard images (camera coordinate frame)
image_points = []

# Set termination criteria. We stop either when an accuracy is reached or when
# we have finished a certain number of iterations.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define real world coordinates for points in the 3D coordinate frame
# Object points are (0,0,0), (1,0,0), (2,0,0) ...., (5,8,0)
object_points_3D = np.zeros((nX * nY, 3), np.float32)

# These are the x and y coordinates
object_points_3D[:, :2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2)

object_points_3D = object_points_3D * square_size


def chessboard_calibration(images_path, calibration_file, resolution, camera, save_images=False):
    # Get the file path for images in the current directory
    print("Start Chessboard calibration...")
    images = [img for img in glob.glob(images_path + '/*.jpg') if not img.endswith('_calib.jpg')]

    print(f"Performing calibration on {len(images)} images")

    # Go through each chessboard image, one by one
    for idx, image_file in enumerate(images):

        # Load the image
        image = cv2.imread(image_file)

        print(idx, end=" ")

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the corners on the chessboard
        success, corners = cv2.findChessboardCorners(gray, (nY, nX), None)

        # If the corners are found by the algorithm, draw them
        if success:
            # Append object points
            object_points.append(object_points_3D)

            # Find more exact corner pixels
            corners_2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Append image points
            image_points.append(corners_2)

            # Draw the corners
            cv2.drawChessboardCorners(image, (nY, nX), corners_2, success)
            #
            # # Display the image. Used for testing.
            # cv2.imshow("Image", image)
            image_file_name, _ = splitExtension(image_file)
            cv2.imwrite(image_file_name + "_calib.jpg", image)
            #
            # # Display the window for a short period. Used for testing.
            # cv2.waitKey(200)

    # Get the dimensions of the image
    height, width = cv2.imread(images[0]).shape[:2]

    # Perform camera calibration to return the camera matrix, distortion coefficients, rotation and translation
    # vectors etc
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points,
                                                       image_points,
                                                       (width, height),
                                                       None,
                                                       None)

    print(f"Camera Height: {height}, Width: {width}")
    # Refine camera matrix
    # Returns optimal camera matrix and a rectangular region of interest
    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist,
                                                               (width, height),
                                                               1,
                                                               (width, height))

    # Crop the image. Uncomment these two lines to remove black lines
    # on the edge of the undistorted image.
    # x, y, w, h = roi
    # undistorted_image = undistorted_image[y:y+h, x:x+w]

    # Display key parameter outputs of the camera calibration process
    #print("Optimal Camera matrix:")
    #print(optimal_camera_matrix)

    #print("\n Distortion coefficient:")
    #print(dist)

    #print("\n Rotation Vectors:")
    #print(rvecs)

    #print("\n Translation Vectors:")
    #print(tvecs)

    pass
    # print("Camera matrix is \n", mtx,
    #       "\n And is stored in " + path + filename + " file along with distortion coefficients : \n", dist)
    data = {'resolution': [resolution[0], resolution[1]],
            'camera': camera,
            'camera_matrix': np.asarray(mtx).tolist(),
            'dist_coeff': np.asarray(dist).tolist()}

    print(calibration_file)
    with open(calibration_file, "w") as f:
        yaml.dump(data, f)

    print("Chessboard calibration complete")
    # return path + filename
    # Save the undistorted image
    #cv2.imwrite(new_filename, undistorted_image)

    # Close all windows
    #cv2.destroyAllWindows()
