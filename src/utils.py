from os import listdir
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML
import matplotlib.animation as animation
from torch import mul


def local_to_global(p, q, local_point):
    local_rotated = quaternion.rotate_vectors(q, local_point)

    return local_rotated + p


def global_to_local(p, q, global_point):

    translated_point = global_point - p

    q_inverse = np.quaternion.conj(q)

    local_point = quaternion.rotate_vectors(q_inverse, translated_point)

    return local_point

def calculate_focal_length(fov_degrees, image_width):

    fov_radians = np.deg2rad(fov_degrees)
    focal_length = (image_width / 2) / np.tan(fov_radians / 2)
    return focal_length

def annotate_image_offline(annotation, image, fov):
            
    local_point = annotation['curr_local_coords']
    point_3d = [local_point[0], -local_point[1], -local_point[2]] #inconsistency between habitat camera frame and classical convention
    focal_length = calculate_focal_length(fov, image.shape[1])
    x = focal_length * point_3d[0] / point_3d[2]
    x_pixel = int(image.shape[1] / 2 + x)

    y =  focal_length * point_3d[1] / point_3d[2]
    y_pixel = int(image.shape[0] / 2 + y)
    label = annotation['obj']
    # Assuming you have an image captured from the sensor
    cv2.circle(image, (x_pixel, y_pixel), 5, (255, 0, 0), -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    font_thickness = 1
    text_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_x = int(x_pixel - text_size[0] // 2)
    text_y = int(y_pixel + text_size[1] + 15)
    rect_top_left = (text_x-3, text_y - text_size[1])  # Top-left corner
    rect_bottom_right = (text_x + text_size[0], text_y + 3)  # Bottom-right corner

    # Draw the rectangle to highlight the background
    cv2.rectangle(image, rect_top_left, rect_bottom_right, (255, 255, 255), -1)
    cv2.putText(image, label, (text_x, text_y), font, font_scale, font_color, font_thickness)

    return image

def plot_correlation_scatter(df, xvar, yvar):
    correlation = df[xvar].corr(df[yvar])

    plt.figure(figsize=(10, 6))
    plt.scatter(df[xvar], df[yvar], alpha=0.5)
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.title(f'Scatter of {xvar} vs {yvar}')

    # Add the correlation label
    plt.text(0.1, max(df[yvar]) * 0.9, f'Correlation: {correlation:.2f}', fontsize=12, ha='left')
    plt.show()

def plot_distribution(df, var, bins=20):
    mean_value = df[var].mean()
    median_value = df[var].median()
    
    plt.figure(figsize=(10, 6))
    plt.hist(df[var], bins=bins, color='blue', alpha=0.7)
    
    # Plot mean and median lines
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_value:.2f}')
    
    # Add labels and title
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {var}')
    
    plt.legend()
    plt.show()

def plot_heatmap(df, xvar, yvar, value):
    heatmap_data = df.pivot_table(index=xvar, columns=yvar, values=value, aggfunc='mean')

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', cbar_kws={'label': value})
    plt.title('Heatmap of Weighted Accuracy')
    plt.show()

def plot_groupby(df, groupby, var, std=True, title=''):
    # df['group'] = df['itr'] // 50

    # Group by the new 'group' column
    grouped = df.groupby(groupby)[var]

    # Calculate mean and standard deviation for each group
    mean_accuracy = grouped.mean()
    std_accuracy = grouped.std()

    # Create a bar plot with error bars
    plt.figure(figsize=(10, 6))
    if std:
        plt.bar(range(len(mean_accuracy)), mean_accuracy, yerr=std_accuracy, capsize=5)
        plt.xticks(range(len(mean_accuracy)), mean_accuracy.index)
        plt.title(f'Mean ± 1 SD of {var} vs groupby for {title}')   
    else:
        plt.bar(range(len(mean_accuracy)), mean_accuracy)
        plt.xticks(range(len(mean_accuracy)), mean_accuracy.index)
        plt.title(f'Mean of {var} vs groupby for {title}')   

    plt.xlabel(groupby)
    plt.ylabel(var)
    plt.show()

def line_plot(df, column):

    plt.figure(figsize=(10, 6))
    
    # Plot the column values over time
    plt.plot(df.index, df[column], marker='o', linestyle='-', label=column)

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.title(f'{column} Over Time')

    plt.grid(True)
    plt.legend()
    plt.show()

def plot_trajectory(df):

    coordinates = np.array(df['agent_location'].tolist())
    x = coordinates[:, 2]
    y = coordinates[:, 1]
    z = coordinates[:, 0]

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, z, c=y, cmap='viridis', s=50, label='Agent Location')

    # Plot arrows to show the direction of movement
    plt.quiver(x[:-1], z[:-1], x[1:] - x[:-1], z[1:] - z[:-1], angles='xy', scale_units='xy', scale=1, color='gray', alpha=0.5)

    # Add a colorbar
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Y Coordinate')

    # Label start and end points
    plt.text(x[0], z[0], 'Start', fontsize=12, color='green', ha='right')
    plt.text(x[-1], z[-1], 'End', fontsize=12, color='red', ha='left')

    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Z Coordinate')
    plt.title('Agent Trajectory Over Time')

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_results(df, run_name):

    df['x_weighted_accuracy'] = df['x_pts_weighted'] / df['x_possible_pts_weighted']
    df['y_weighted_accuracy'] = df['y_pts_weighted'] / df['y_possible_pts_weighted']
    df['z_weighted_accuracy'] = df['z_pts_weighted'] / df['z_possible_pts_weighted']
    df['overall_weighted_accuracy'] = (df['x_pts_weighted'] + df['y_pts_weighted'] + df['z_pts_weighted']) / (df['x_possible_pts_weighted'] + df['y_possible_pts_weighted'] + df['z_possible_pts_weighted'])

    labels = ['x_weighted_accuracy', 'y_weighted_accuracy', 'z_weighted_accuracy', 'overall_weighted_accuracy']
    values = [
        df['x_weighted_accuracy'].mean(),
        df['y_weighted_accuracy'].mean(),
        df['z_weighted_accuracy'].mean(),
        df['overall_weighted_accuracy'].mean()
    ]

    # Calculate the standard deviations
    errors = [
        df['x_weighted_accuracy'].std(),
        df['y_weighted_accuracy'].std(),
        df['z_weighted_accuracy'].std(),
        df['overall_weighted_accuracy'].std()
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, yerr=errors, color=['blue', 'green', 'red', 'purple'], alpha=0.7, capsize=10)
    plt.xlabel('Metrics')
    plt.ylabel('Weighted Accuracy')
    plt.title('Weighted Accuracy for x, y, z and Overall')
    plt.show()

def gif(path):
    fig = plt.figure()
    ims = []

    for i in range(len(listdir(path))):
        try:
            ndx=0
            if len(listdir(f"{path}/step{i}")) > 3:
                ndx=1
                
            np_img_i = cv2.imread(f"{path}/step{i}/image{ndx}.png")

            np_img_i = cv2.cvtColor(np_img_i, cv2.COLOR_BGR2RGB)

            im = plt.imshow(np_img_i)
            ims.append([im])
            np_img_i_copy = cv2.imread(f"{path}/step{i}/copy_image{ndx}.png")
            np_img_i_copy = cv2.cvtColor(np_img_i_copy, cv2.COLOR_BGR2RGB)
            im2 = plt.imshow(np_img_i_copy)
            ims.append([im2])

        except Exception as e:
            print(f"Image step{i} not found")
            continue

    ani = animation.ArtistAnimation(fig, ims, interval=400, blit=True)
    HTML(ani.to_jshtml())
    ani.save(f'{path}/animagion.gif', writer='imagemagick')

