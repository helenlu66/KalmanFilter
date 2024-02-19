import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_errors(errors:list, categories:list):
    
    # Creating the bar plot
    plt.figure(figsize=(10, 6))  # Set the figure size as needed
    plt.bar(categories, errors, color='skyblue')  # Create a bar plot

    # Adding title and labels
    plt.title('Error Distance Bar Plot')
    plt.xlabel('Sensor Fail Probability')  # Adjust as needed
    plt.ylabel('Distance from true position')

    # Adding error values on top of each bar
    for i, v in enumerate(errors):
        plt.text(i, v - 1, v, ha='center', va="top")

    # Saving the figure to a PNG file
    plt.savefig('error_distance_bar_plot.png', bbox_inches='tight')  # Adjust the path and filename as needed
    plt.close()  # Close the plotting window

def plot_ellipse(mean, cov, filename="uncertainty_ellipse"):
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Calculate the angle of rotation of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
    
    # Width and height of the ellipse are sqrt of the eigenvalues
    width, height = 2 * np.sqrt(eigenvalues)
    
    # Create an ellipse
    ellipse = Ellipse(xy=mean.squeeze(), width=width, height=height, angle=angle, edgecolor='r', fc='None', lw=2)
    
    # Plot the ellipse
    fig, ax = plt.subplots()
    ax.add_patch(ellipse)
    
    # Set plot limits
    ax = plt.gca()
   
    plt.xlim(mean[0] - 7, mean[0] + 7)
    plt.ylim(mean[1] - 5, mean[1] + 5)
    
    # Plot the mean
    plt.plot(mean[0], mean[1], 'ro') # 'ro' plots a red dot
    
    # Display the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Uncertainty ellipse representing 1 standard deviation')
    #plt.grid(True)
    #plt.axis('equal') # Equal aspect ratio ensures that the ellipse is not distorted
    # Save the plot to a file
    plt.savefig(filename, bbox_inches='tight')
    plt.close() # Close the plot to free memory

def mahalanobis_ellipse_points(mean, cov, num_points=100):
    # Ensure the covariance matrix is a numpy array
    cov = np.array(cov)
    mean = np.array(mean)
    
    # Invert the covariance matrix
    cov_inv = cov
    
    # Eigen decomposition of the inverse covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_inv)
    
    # Angle for parameterization
    t = np.linspace(0, 2 * np.pi, num_points)
    
    # Ellipse in standard position
    ellipse_std = np.array([np.sqrt(eigenvalues[0]) * np.cos(t), np.sqrt(eigenvalues[1]) * np.sin(t)])
    
    # Rotate and translate ellipse to original position
    ellipse = np.dot(eigenvectors, ellipse_std).T + mean
    
    return ellipse

def plot_and_save_ellipse(mean, cov, filename="ellipse_plot.png"):
    points = mahalanobis_ellipse_points(mean, cov)
    
    plt.figure(figsize=(8, 6))
    plt.plot(points[:,0], points[:,1], 'b-', label="1 Mahalanobis Distance") # Plot ellipse
    plt.plot(mean[0], mean[1], 'ro', label="Mean") # Plot mean
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points 1 Mahalanobis Distance Away')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    # Save the plot to a file
    plt.savefig(filename, bbox_inches='tight')
    plt.close() # Close the plot to free memory


def mahalanobis_distance(point, mean, cov):
    # Convert inputs to numpy arrays for matrix operations
    point = np.array(point)
    mean = np.array(mean)
    cov = np.array(cov)
    
    # Calculate the difference between the point and the mean
    diff = point - mean
    
    # Invert the covariance matrix
    cov_inv = np.linalg.inv(cov)
    
    # Calculate the Mahalanobis distance using the `@` operator for matrix multiplication
    distance = np.sqrt(diff.T @ cov_inv @ diff)
    
    return distance
# Example usage
