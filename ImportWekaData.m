% Import Weka data for manipulation
javaaddpath('C:\Users\Laurel\Matlab\CS534\src\cs534_git\FinalProject\weka.jar');
clear all
clc

%import java.io.*;
fName = 'C:\Users\Laurel\Matlab\CS534\src\cs534_git\FinalProject\Merged_Weka_Input.arff';

%## read file
loader = weka.core.converters.ArffLoader();
loader.setFile( java.io.File(fName) );
D = loader.getDataSet();
D.setClassIndex( D.numAttributes()-1 );

%## dataset
relationName = char(D.relationName);
numAttr = D.numAttributes;
numInst = D.numInstances;

%## attributes
%# attribute names
attributeNames = arrayfun(@(k) char(D.attribute(k).name), 0:numAttr-1, 'Uni',false);

%# attribute types
types = {'numeric' 'nominal' 'string' 'date' 'relational'};
attributeTypes = arrayfun(@(k) D.attribute(k-1).type, 1:numAttr);
attributeTypes = types(attributeTypes+1);

%# nominal attribute values
nominalValues = cell(numAttr,1);
for i=1:numAttr
    if strcmpi(attributeTypes{i},'nominal')
        nominalValues{i} = arrayfun(@(k) char(D.attribute(i-1).value(k-1)), 1:D.attribute(i-1).numValues, 'Uni',false);
    end
end

%instances
data = zeros(numInst,numAttr);
for i=1:numAttr
    data(:,i) = D.attributeToDoubleArray(i-1);
    
end
for i=1:numInst
    data(i,numAttr) = D.classAttribute()
end

fdata = filteredZeros(data, numAttr, numInst);
meanVector = zeros(numInst,1);
[fdata, meanVector] = filteredMean(data, meanVector, numAttr, numInst);

trainMean = [];
testMean = [];
percent = 30;
[trainMean, testMean] = splitData(meanVector, testMean, trainMean, numInst, percent);

%Output file for weak
csvwrite('trainAverages.csv', trainMean)
csvwrite('testAverages.csv', testMean)

%convert outliers to mean:
%go thru data row by row
function [data, meanVector] = filteredMean(data, meanVector, numAttr, numInst)
    for i=1:numInst
        rowMean = 0;
        n = 0;
        %calculate filtered mean for row i
        for j=1:numAttr
            %test if we need to remove 0's
            if data(i,j) < 100 %&& data(i,j) > 0
                rowMean = rowMean + data(i,j);
                n = n+1;       
            end
            
        end
        
        rowMean = rowMean / n;       %filtered avg
        %update mean vector 
        %meanVector(i, 1) = rowMean;
       
        %meanVector(i, 2) = 
        %replace row i's outliers with mean
        for j=1:numAttr
            if data(i,j) > 100
                data(i,j) = rowMean;
            end
        end
    end  
end


%replaces all magnitudes greater than 100 w/zero
function data = filteredZeros(data, numAttr, numInst)
	for i=1:numInst
        for j=1:numAttr
            if data(i,j) > 100
                data(i,j) = 0;            
            end
        end
    end
end

%splits data into test data and train data, percent is percentage of 
% data desired for testing (percent = 30, 30% is test data)
function [testMean, trainMean] = splitData(data, testMean, trainMean, numInst, percent)
    numTest = (percent/100)*numInst;
    %r = randi([1 int32(numInst)], 1, int32(numTest));
    r = randperm(int32(numInst), int32(numTest));
    for i=1:numInst
        %if we do not find i in r
       if sum(find(i==r)) == 0
            %testData = [testData; data(i,:)];
            testMean = [testMean;data(i)];
       else
           %trainData = [trainData; data(i,:)];
           trainMean = [trainMean;data(i)];
       end
    end
end
