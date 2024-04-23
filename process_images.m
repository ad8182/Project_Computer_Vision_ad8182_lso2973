function process_images

    %
    % process_images() is a script meant to be exclusively
    % used with this project. It goes through all of the training
    % images in the NAZCA_SCANNED_GEMS subdirectory, identifying
    % all shapes that look like a rock, and attempts to create a
    % 672x672 image of that rock. This is so that the rock can be
    % used with the resnet50() function in the main project itself.
    %
    % Author: Lily O'Carroll <lso2973>
    % Date: 23 April, 2024
    %

    % This directory contains all of the images
    % that we need to adjust accordingly
    image_dir = "NAZCA_SCANNED_GEMS";

    % Get all of the image files so we can process them
    filelist = dir(fullfile(image_dir, '**\*.jpg'));
    % Remove everything that isn't an image (mainly directories)
    filelist = filelist(~[filelist.isdir]);

    % Go through all of the images in our testing set.
    for i = 1 : size(filelist)
        % Get the filename of the image, read it in,
        % and convert it into a binarized image.
        gem_location = "" + filelist(i).folder + '\' + filelist(i).name;
        im_preprocessed = im2double(imread(gem_location));
        im_gray = rgb2gray(im_preprocessed);
        im_bw = imbinarize(im_gray);
    
        % Set up disks for dilation (disks A, C) and
        % a disk for erosion (disk B).
        disk_A = strel("disk", 5);
        disk_B = strel("disk", 10);
        disk_C = strel("disk", 6);
        
        % Clean up the image the best we can
        im_dilated = imdilate(im_bw, disk_A);
        im_eroded = imerode(im_dilated, disk_B);
        im_binary = imdilate(im_eroded, disk_C);
    
        % Do connected component analysis to get the region where
        % the rock is.
        [im_connected_components, number_of_cc] = bwlabel(~im_binary, 4);
        
        % Largest amount of pixels should be the rock
        highest_num_of_pixels = 0;
        % Top-left corner of the largest rock region
        rock_x = 0;
        rock_y = 0;
    
        % For all components found in this image,
        for this_component = 1 : number_of_cc

            % Get the binary image, and how many pixels
            % it consists of
            binary_image = (im_connected_components == this_component);
            n_pix = sum(binary_image(:));
            % Set up regionprops() to get the bounding box
            ss = regionprops(binary_image, 'BoundingBox');

            % If the number of pixels in this binary region is
            % the highest amount found so far,

            %%% IDEA -- Use the second-highest amount of pixels?
            %%% this accounts for the fact that this script only works
            %%% well on darker-colored gemstones, and will often
            %%% miss the lighter-colored gemstones in some capacity.
            %%% For example, this script COMPLETELY fails with
            %%% the moonstone dataset...
            if (n_pix > highest_num_of_pixels)
                % Set the highest number of pixels, and
                % the "bounding box" points
                highest_num_of_pixels = n_pix;
                bounding_box_points = ss.BoundingBox;
                rock_x = bounding_box_points(1);
                rock_y = bounding_box_points(2);
            end
        end
    
        % Grab the final image, using the "bounding box"
        % points found earlier to crop it for resnet50()
        % (This is why the image size is 672x672 -- it
        % will be shrunken down to 224x224)
        final_image = imcrop(im_preprocessed, ... 
            [(rock_x-84) (rock_y-84) 672 672]);

        % Write the image in that file location
        imwrite(final_image, gem_location);

    end

end