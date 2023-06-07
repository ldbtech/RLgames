import numpy as np
import cv2


class SIFT:
    def __init__(self, image):
        self.image = image

    # Step 1: Gaussian scale space
    def build_gaussian_pyramid(self, image, num_octaves, num_scales):
        # Implement Gaussian pyramid construction
        # Return the Gaussian pyramid
        gaussian_pyramid = []

        for oct in range(num_octaves):
            octave_images = []
            for scale in range(num_scales):
                # We use downsmapling  (Scale factor for downsampling)
                scale_factor = 2 ** (oct + scale / num_scales)

                # Resize the image
                resised_image = cv2.resize(
                    image,
                    None,
                    fx=1 / scale_factor,
                    fy=1 / scale_factor,
                    interpolation=cv2.INTER_LINEAR,
                )

                # Apply the Gaussian Smoothing
                sigma = 1.6 * scale_factor  # Standard Deviation for Gaussian Kernel
                smooth_image = cv2.GaussianBlur(
                    resised_image, (0, 0), sigmaX=sigma, sigmaY=sigma
                )

                # Append to the octave_images
                octave_images.append(smooth_image)

            gaussian_pyramid.append(octave_images)

            # downsample the image to the next octave
            resised_image = cv2.resize(
                image,
                None,
                fx=1 / scale_factor,
                fy=1 / scale_factor,
                interpolation=cv2.INTER_LINEAR,
            )

        return gaussian_pyramid

    # Step 2: Difference of Gaussians (DoG)
    def build_difference_of_gaussians(pyramid):
        # Implement DoG pyramid construction
        # Return the DoG pyramid
        length = len(pyramid)
        dog = []

        for octave_imgs in pyramid:
            octave_dog = []
            # iterate over the scales within octaves
            for i in range(len(octave_imgs) - 1):
                dog_image = octave_imgs[i + 1] - octave_imgs[i]
                octave_dog.append(dog_image)
            dog.append(octave_dog)

        return dog

    # Step 3: Keypoint detection
    def is_extremum(self, pixel_value, dog_prev, dog_current, dog_next, row, col):
        # Placeholder function - Implement the extremum test logic here
        # Compare the pixel_value with its 26 neighbors in the current, previous, and next DoG images
        # Return True if the pixel_value is an extremum, False otherwise
        return False

    def compute_response(self, pixel_value, dog_prev, dog_current, dog_next, row, col):
        # Placeholder function - Implement the response computation logic here
        # Compute the response value (e.g., intensity difference) for the pixel
        # based on its value and the neighboring pixels in the DoG images
        return 0.1

    def detect_keypoints(self, dog_pyramid, threshold):
        # Implement keypoint detection in the DoG pyramid
        # Return the detected keypoints
        keypoints = []
        for octave_dog in dog_pyramid:
            octave_keypoints = []
            for i in range(1, len(octave_dog) - 1):
                dog_current = octave_dog[i]
                dog_prev = octave_dog[i - 1]
                dog_next = octave_dog[i + 1]
                for row in range(1, dog_current.shape[0] - 1):
                    for col in range(1, dog_current[1] - 1):
                        pixel_value = dog_current[row, col]

                        # Perform extremum test
                        if self.is_exteremum(
                            pixel_value, dog_prev, dog_current, dog_next, row, col
                        ):
                            # Compute Intensity difference
                            response = self.compute_response(
                                pixel_value, dog_prev, dog_current, dog_next, row, col
                            )
                            if response > threshold:
                                keypoint = (row, col, i)
                                octave_keypoints.append(keypoint)
            keypoints.append(octave_keypoints)

        return keypoints

    # Step 4: Keypoint orientation assignment
    def assign_keypoint_orientations(self, keypoints, gaussian_pyramid):
        # Implement keypoint orientation assignment
        # Return the keypoints with assigned orientation
        keypoint_ = []
        num_bins = 36
        angel_step = 2 * np.pi / num_bins
        for keypoint in keypoints:
            octave_idx, scale, row, col = keypoint
            # Get gaussian image for the keypoint's octave and scale
            gaussian_image = gaussian_pyramid[octave_idx][scale]
            # Compute the orientation histogram
            orientation_histogram = np.zeros(num_bins)

            for i in range(-8, 9):
                for j in range(-8, 9):
                    # Calculate the gradient magnititude and angle
                    dx = (
                        gaussian_image[row + i, col + j + 1]
                        - gaussian_image[row + i, col + j - 1]
                    )
                    dy = (
                        gaussian_image[row + i + 1, col + j]
                        - gaussian_image[row + i - 1, col + j]
                    )
                    magnititude = np.sqrt(dx**2 + dy**2)
                    angle = np.arctan2(dy, dx)

                    # Convert the angle into range of [0, 2pi]
                    if angle < 0:
                        angle += 2 * np.pi

                    # Calculate the bin index for the orientation histogram
                    bin_index = int(angle // angel_step)
                    # Acumulate the gradient magnititude in the corresponding bin
                    orientation_histogram[bin_index] += magnititude

            # Find the dominance orientation in the histogram
            # Find the dominant orientation(s) in the histogram
            max_magnitude = np.max(orientation_histogram)
            dominant_orientations = []

            for i in range(num_bins):
                if orientation_histogram[i] >= 0.8 * max_magnitude:
                    # Convert the bin index to the corresponding orientation angle
                    angle = (i + 0.5) * angel_step
                    dominant_orientations.append(angle)

            # Assign the dominant orientation(s) to the keypoint
            keypoint_.append(dominant_orientations)
        return keypoint_

    # Step 5: Keypoint descriptor computation
    def compute_keypoint_descriptors(self, keypoints, gaussian_pyramid):
        descriptor_size = 16  # Size of the descriptor patch (e.g., 16x16)
        descriptor_scale = 3  # Scaling factor for the descriptor patch
        descriptor_orientation_bins = 8  # Number of orientation bins in the descriptor

        # Iterate over the keypoints
        for keypoint in keypoints:
            octave_idx, scale, row, col, orientations = keypoint

            # Get the Gaussian image for the keypoint's octave and scale
            gaussian_image = gaussian_pyramid[octave_idx][scale]

            # Compute the descriptor
            descriptor = np.zeros(
                (descriptor_size, descriptor_size, descriptor_orientation_bins)
            )

            for i in range(-descriptor_size // 2, descriptor_size // 2):
                for j in range(-descriptor_size // 2, descriptor_size // 2):
                    # Rotate the descriptor patch based on the keypoint orientation(s)
                    for orientation in orientations:
                        # Compute the rotated coordinates within the descriptor patch
                        sin_theta = np.sin(orientation)
                        cos_theta = np.cos(orientation)
                        new_i = (cos_theta * i - sin_theta * j) / descriptor_scale
                        new_j = (sin_theta * i + cos_theta * j) / descriptor_scale

                        # Translate the rotated coordinates to the image coordinates
                        x = col + int(np.round(new_j))
                        y = row + int(np.round(new_i))

                        # Calculate the gradient magnitude and angle at the rotated position
                        dx = gaussian_image[y, x + 1] - gaussian_image[y, x - 1]
                        dy = gaussian_image[y + 1, x] - gaussian_image[y - 1, x]
                        magnitude = np.sqrt(dx**2 + dy**2)
                        angle = np.arctan2(dy, dx)

                        # Convert the angle to the range [0, 2pi]
                        if angle < 0:
                            angle += 2 * np.pi

                        # Calculate the bin index for the orientation histogram
                        bin_index = int(
                            angle * descriptor_orientation_bins / (2 * np.pi)
                        )

                        # Accumulate the gradient magnitude in the corresponding bin
                        descriptor[
                            i + descriptor_size // 2,
                            j + descriptor_size // 2,
                            bin_index,
                        ] += magnitude

            # Normalize the descriptor
            descriptor /= np.linalg.norm(descriptor)

            # Threshold the descriptor values to 0.2 and re-normalize
            descriptor[descriptor > 0.2] = 0.2
            descriptor /= np.linalg.norm(descriptor)

            # Flatten the descriptor and assign it to the keypoint
            keypoint.append(descriptor.flatten())

        return keypoints


# Main function
def main():
    # Load input image
    image = cv2.imread("input_image.jpg", 0)
    sift = SIFT(image=image)

    # Step 1: Gaussian scale space
    gaussian_pyramid = sift.build_gaussian_pyramid(image, num_octaves=4, num_scales=5)

    # Step 2: Difference of Gaussians (DoG)
    dog_pyramid = sift.build_difference_of_gaussians(gaussian_pyramid)

    # Step 3: Keypoint detection
    keypoints = sift.detect_keypoints(dog_pyramid, threshold=0.1)

    # Step 4: Keypoint orientation assignment
    keypoints_with_orientations = sift.assign_keypoint_orientations(
        keypoints, gaussian_pyramid
    )

    # Step 5: Keypoint descriptor computation
    descriptors = sift.compute_keypoint_descriptors(
        keypoints_with_orientations, gaussian_pyramid
    )

    # Display keypoints and descriptors
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
    cv2.imshow("Keypoints", image_with_keypoints)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
