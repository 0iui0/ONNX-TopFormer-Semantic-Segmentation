import cv2
import numpy as np
import pafy
import pyrealsense2 as rs

from topformer import TopFormer
def main():
	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()

	# Get device product line for setting a supporting resolution
	pipeline_wrapper = rs.pipeline_wrapper(pipeline)
	pipeline_profile = config.resolve(pipeline_wrapper)
	device = pipeline_profile.get_device()
	device_product_line = str(device.get_info(rs.camera_info.product_line))

	found_rgb = False
	for s in device.sensors:
		if s.get_info(rs.camera_info.name) == 'RGB Camera':
			found_rgb = True
			break
	if not found_rgb:
		print("The demo requires Depth camera with Color sensor")
		exit(0)

	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

	if device_product_line == 'L500':
		config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
	else:
		config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

	# Start streaming
	pipeline.start(config)

	# Initialize semantic segmentator
	model_path = "models/TopFormer-S_512x512_2x8_160k.onnx"
	segmentator = TopFormer(model_path)

	cv2.namedWindow("Semantic Sementation", cv2.WINDOW_NORMAL)
	try:
		while True:

			# Wait for a coherent pair of frames: depth and color
			frames = pipeline.wait_for_frames()
			color_frame = frames.get_color_frame()
			if not color_frame:
				continue

			# Convert images to numpy arrays
			color_image = np.asanyarray(color_frame.get_data())
			color_colormap_dim = color_image.shape

			# Update semantic segmentator
			seg_map = segmentator(color_image)
			combined_img = segmentator.draw_segmentation(color_image, alpha=0.5)
			cv2.imshow("Semantic Sementation", combined_img)

			# # Show images
			# cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
			# cv2.imshow('RealSense', images)
			cv2.waitKey(1)

	finally:

		# Stop streaming
		pipeline.stop()
if __name__ == '__main__':
		main()
