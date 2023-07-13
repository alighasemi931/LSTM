clc
clear all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the sensor data

load('Zinq_Log_INS_SemiStatic7_AI.mat');

%calculate for X_AXIS
sensor1DataX = pkt_Data_AI(1:10000,1)  ;
sensor2DataX = pkt_Data_AI(1:10000,2)  ;
sensor3DataX = pkt_Data_AI(1:10000,3)  ;
refDataX     = pkt_Data_AI(1:10000,4)  ;

%calculate for Y_AXIS
sensor1DataY = pkt_Data_AI(1:10000,5)  ;
sensor2DataY = pkt_Data_AI(1:10000,6)  ;
sensor3DataY = pkt_Data_AI(1:10000,7)  ;
refDataY     = pkt_Data_AI(1:10000,8)  ;

%calculate for Z_AXIS
sensor1DataZ = pkt_Data_AI(1:10000,9)   ;
sensor2DataZ = pkt_Data_AI(1:10000,10)  ;
sensor3DataZ = pkt_Data_AI(1:10000,11)  ;
refDataZ     = pkt_Data_AI(1:10000,12)  ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the input layer and LSTM layer

inputSize = size(sensor1DataX,1);
numHiddenUnits = 50;
num_layers = 3;
numClasses = inputSize ;
layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    %repmat(lstmLayer(numHiddenUnits,'OutputMode','sequence'),num_layers-1,1)
    fullyConnectedLayer(numClasses)
    regressionLayer];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set the training options for the network
options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'MiniBatchSize',64, ...
    'InitialLearnRate',0.001, ...
    'SequenceLength',50, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment','gpu',...
    'L2Regularization' , 0.01);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Train the network

netX = trainNetwork(refDataX, refDataX, layers, options);
netY = trainNetwork(refDataY, refDataY, layers, options);
netZ = trainNetwork(refDataZ, refDataZ, layers, options);

% Remove noise from the gyroscope data

denoisedDataX_1 = predict(netX, sensor1DataX);
denoisedDataX_2 = predict(netX, sensor2DataX);
denoisedDataX_3 = predict(netX, sensor3DataX);
%%calculate max  similarity
similarityX_1 = corr(refDataX, denoisedDataX_1 );
similarityX_2 = corr(refDataX, denoisedDataX_2 );
similarityX_3 = corr(refDataX, denoisedDataX_3 );

[~,index ] =max([similarityX_1,similarityX_2,similarityX_3]);
if index==1
denoisedDataX = denoisedDataX_1 ;
elseif index==2
denoisedDataX = denoisedDataX_2 ;
else 
denoisedDataX = denoisedDataX_3 ;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
denoisedDataY_1 = predict(netY, sensor1DataY);
denoisedDataY_2 = predict(netY, sensor2DataY);
denoisedDataY_3 = predict(netY, sensor3DataY);

%%calculate max  similarity
similarityY_1 = corr(refDataY, denoisedDataY_1 );
similarityY_2 = corr(refDataY, denoisedDataY_2 );
similarityY_3 = corr(refDataY, denoisedDataY_3 );

[~,index ] =max([similarityY_1,similarityY_2,similarityY_3]);
if index==1
denoisedDataY = denoisedDataY_1;
elseif index==2
denoisedDataY = denoisedDataY_2;
else 
denoisedDataY = denoisedDataY_3;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
denoisedDataZ_1 = predict(netZ, sensor1DataZ);
denoisedDataZ_2 = predict(netZ, sensor2DataZ);
denoisedDataZ_3 = predict(netZ, sensor3DataZ);
%%calculate max  similarity
similarityZ_1 = corr(refDataZ, denoisedDataZ_1 );
similarityZ_2 = corr(refDataZ, denoisedDataZ_2 );
similarityZ_3 = corr(refDataZ, denoisedDataZ_3 );

[~,index ] =max([similarityZ_1,similarityZ_2,similarityZ_3]);
if index==1
denoisedDataZ = denoisedDataZ_1;
elseif index==2
denoisedDataZ = denoisedDataZ_2;
else 
denoisedDataZ = denoisedDataZ_3;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the RMSE between the original and denoised data

rmseX = sqrt(mean(refDataX - denoisedDataX).^2);
rmseY = sqrt(mean(refDataY - denoisedDataY).^2);
rmseZ = sqrt(mean(refDataZ - denoisedDataZ).^2);

% Print the RMSE
fprintf( 'RMSE on X_AXIS: %f\n' , rmseX );
fprintf( 'RMSE on Y_AXIS: %f\n' , rmseY );
fprintf( 'RMSE on Z_AXIS: %f\n' , rmseZ );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%store DATA respectively :1-sensorData 2-refDATA 3-denisedDATA on X_AXIS
%Y_AXIS and Z_AXIS :9dimensions

dataX = [ sensor1DataX , sensor2DataX , sensor3DataX , denoisedDataX , refDataX ];
dataY = [ sensor2DataY , sensor2DataY , sensor3DataY , denoisedDataY , refDataY ];
dataZ = [ sensor1DataZ , sensor1DataZ , sensor3DataZ , denoisedDataZ , refDataZ ];
data =  [ dataX , dataY , dataZ ];

save('gyro_data.mat','data');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Plot the original and denoised data on X_AXIS
figure(1);
plot(refDataX(1:100000, :), 'LineWidth', 1);
hold on;
plot(denoisedDataX(1:100000, :), 'LineWidth', 1);
legend('refDataX', 'DenoisedX');
xlabel('Sample');
ylabel('ref Data');
title('Comparison of Original and Denoised Sensor Data on X_AXIS');


% Plot the original and denoised data on Y_AXIS
figure(2); 
plot(refDataY(1:100000, :), 'LineWidth', 1);
hold on;
plot(denoisedDataY(1:100000, :), 'LineWidth', 1);
legend('refDataY', 'DenoisedY');
xlabel('Sample');
ylabel('ref Data');
title('Comparison of Original and Denoised Sensor Data on Y_AXIS');

% Plot the original and denoised data on Z_AXIS
figure(3);
plot(refDataZ(1:100000, :), 'LineWidth', 1);
hold on;
plot(denoisedDataZ(1:100000, :), 'LineWidth', 1);
legend('refDataZ', 'DenoisedZ');
xlabel('Sample');
ylabel('ref Data');
title('Comparison of Original and Denoised Sensor Data on Z_AXIS');
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%