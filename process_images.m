function process_images

    %
    % process_images() is a script meant to be exclusively
    % used with this project. It goes through all of the training
    % images in the NAZCA_SCANNED_GEMS subdirectory, identifying
    % all shapes that look like a rock, and attempts to create a
    % 672x672 image of that rock. It then writes that cropped image in
    % the OUTPUT folder. This is so that the rock can be
    % used with the resnet50() function in the main project itself.
    %
    % Author: Lily O'Carroll <lso2973>
    %         Andrew Dantone <ad8182>
    % Date: 26 April, 2024
    %

    % This is the directory where the cropped and scaled
    % images will be saved to -- this is done so that the
    % original dataset of gemstones doesn't get overwritten.
    image_dir = "OUTPUT";

    % Copy all of the images in the dataset (located in
    % NAZCA_SCANNED_GEMS) to the "OUTPUT" directory.
    copyfile NAZCA_SCANNED_GEMS OUTPUT

    % Get all of the image files so we can process them
    filelist = dir(fullfile(image_dir, '**\*.jpg'));
    % Remove everything that isn't an image (mainly directories)
    filelist = filelist(~[filelist.isdir]);

    % Go through all of the images in our testing set.
    %%% Starting in R2024a this specific line of code will throw
    %%% a warning, despite it working just fine.
    for img = 1 : size(filelist)

        % Get the filename of the image, read it in,
        % and convert it into a binarized image.
        gem_location = "" + filelist(img).folder + '\' ...
            + filelist(img).name;
        im_preprocessed = im2double(imread(gem_location));
        im_gray = rgb2gray(im_preprocessed);
        im_adapt_hist = adapthisteq(im_gray);
        threshold = graythresh(im_adapt_hist);
        im_bw = imbinarize(im_gray, threshold);
        disk_N = strel("disk", 5);
        im_bw = imerode(im_bw, disk_N);
        
        % Since some rocks will be classified as foreground and some as
        % background, we need to find the largest area and assume that's 
        % the background.
        [im_connected_components, number_of_cc] = bwlabel(~im_bw, 4);
        max_pixels = 0;
        max_component = -1;
        for this_component = 0 : number_of_cc
            binary_image = (im_connected_components == this_component);
            n_pix = sum(binary_image(:));
            if (n_pix > max_pixels)
                max_pixels = n_pix;
                max_component = this_component;
            end
        end
        if (max_component == 0)
            im_bw = ~ im_bw;
        end

        % Set up disks for dilation (disks A, C) and
        % a disk for erosion (disk B).
        disk_A = strel("disk", 60);
        disk_B = strel("disk", 60);
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
        seccond_highest_num_of_pixels = 0;
        % Top-left corner of the largest rock region
        highest_rock_x = 0;
        seccond_highest_rock_x = 0;
        highest_rock_y = 0;
        seccond_highest_rock_y = 0;
        % For all components found in this image,
        for this_component = 0 : number_of_cc

            % Get the binary image, and how many pixels
            % it consists of
            binary_image = (im_connected_components == this_component);
            n_pix = sum(binary_image(:));
            % Set up regionprops() to get the bounding box
            ss = regionprops(binary_image, 'BoundingBox');

            % If the number of pixels in this binary region is
            % the second highest amount found so far,
            if (n_pix > highest_num_of_pixels || ... 
                    seccond_highest_num_of_pixels == 0)
                % Set the highest number of pixels, and
                % the "bounding box" points
                %also store the highest and 2nd highest for use later
                if(n_pix > highest_num_of_pixels)
                    seccond_highest_num_of_pixels = highest_num_of_pixels;
                    highest_num_of_pixels = n_pix;
                    bounding_box_points = ss.BoundingBox;
                    seccond_highest_rock_x = highest_rock_x;
                    highest_rock_x = bounding_box_points(1);
                    seccond_highest_rock_y = highest_rock_y;
                    highest_rock_y = bounding_box_points(2);
                else
                    seccond_highest_num_of_pixels = n_pix;
                    bounding_box_points = ss.BoundingBox;
                    seccond_highest_rock_x = bounding_box_points(1);
                    seccond_highest_rock_y = bounding_box_points(2);
                end
            end
        end
    
        % Grab the final image, using the "bounding box"
        % points found earlier to crop it for resnet50()
        % (This is why the image size is 672x672 -- it
        % will be shrunken down to 224x224)
        final_image = imcrop(im_preprocessed, ... 
            [(seccond_highest_rock_x-84) (seccond_highest_rock_y-84) ...
            672 672]);

        % Forcibly resize our image to be 224x224
        % for use with resnetNetwork
        resized_final_image = imresize(final_image, [224 224]);

        % Write the final image to disk, in
        % the new directory (so that the original
        % training set of images doesn't get
        % overwritten)
        imwrite(resized_final_image, gem_location);

    end
    fprintf("Done processing images\n");
end
