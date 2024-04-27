function GemStoneID_lso2973_ad8182

    % 
    % gemstone_ANN uses a re-trained Artificial
    % Neural Network (ANN) in order to identify
    % gemstones. It uses pre-processed images
    % of gemstones in order to learn which gemstones
    % are which. Specifically, it uses MATLAB's
    % resnetNetwork, although it is a retrained
    % version of imagePretrainedNetwork.
    %
    % Author: Lily O'Carroll <lso2973>
    %         Andrew Dantone <ad8182>
    % Date: 26 April, 2024
    %

    % The folder with the training data
    % is the OUTPUT folder.
    folderName = "OUTPUT";

    % Create an imageDatastore of the image
    % data, for use with re-training the
    % imagePretrainedNetwork.
    imds = imageDatastore(folderName, ...
        IncludeSubfolders=true, ...
        LabelSource="foldernames");

    % Get the class names, as well as how
    % many classes there are.
    classNames = categories(imds.Labels);
    numClasses = numel(classNames);

    %%% For testing -- partition the data into training and
    %%% validation sets. 80% of the images will be used for
    %%% training, while the other 20% will for testing.
    [imdsTrain, imdsValidation, imdsTest] = ... 
        splitEachLabel(imds, 0.8, 0.2, "randomized");

    % Set up an artificial neural network, and retrain it
    % to fit our dataset. We tell the neural network that
    % there will be classes equal to the amount of gemstones
    % that we have in our training set.
    ANN = imagePretrainedNetwork(NumClasses=numClasses);

    % Get the neural network input size.
    inputSize = ANN.Layers(1).InputSize;

    % Set the learning rate of both parameters to 10, in order
    % to speed up convergence and the level of updates to this
    % layer.
    ANN = setLearnRateFactor(ANN, "conv10/Weights", 10);
    ANN = setLearnRateFactor(ANN, "conv10/Bias", 10);

    % Set up an Image Datastore for training,
    augimds_train = augmentedImageDatastore(inputSize(1:2), imdsTrain);
    % for validation,
    augimds_validation = augmentedImageDatastore(inputSize(1:2), ...
        imdsValidation);
    % and for testing.
    augimds_test = augmentedImageDatastore(inputSize(1:2), imdsTest);

    % Set up options for our proper training session.
    options = trainingOptions("adam", ...
        InitialLearnRate = 0.0001, ...
        ValidationData = augimds_validation, ...
        ValidationFrequency = 5, ...
        Plots = "training-progress", ...
        Metrics = "accuracy", ...
        Verbose = false, ...
        MaxEpochs = 30);

    % Properly train the artificial neural network.
    ANN = trainnet(augimds_train, ANN, "crossentropy", options);

    %%% NOTE -- For whatever reason, after the first time that this
    %%% was ran, it refuses to work. We don't know why, we don't have
    %%% the time to figure out why, but we were able to at least save
    %%% the general accuracy and the training results of the first (and
    %%% only) successful run of training the ANN.
    YTest = minibatchpredict(ANN, augimds_test);
    YTest = scores2label(YTest, classNames);
    TTest = imdsTest.Labels;

    % Get the general accuracy of our ANN.
    acc = mean(TTest==YTest);

    %%% NOTE -- This was added after our first run, so we're not
    %%% 100% sure that this works. However, the goal is to print
    %%% out each type of gemstone, how many times it was identified
    %%% correctly, and how many times it was misclassified (as
    %%% percentages).
    for class_num = 1 : numClasses
        correct_rate = TTest(class_num)==YTest(class_num);
        fprintf("Class: %s\nCorrect Classification Rate: %d\nIncorrect Classification Rate:%d", ...
            imds.Labels(class_num), correct_rate, 1-correct_rate);

    end
    
    % Print the overall accuracy and end.
    fprintf("Overall Accuracy: %d\n", acc);

end
