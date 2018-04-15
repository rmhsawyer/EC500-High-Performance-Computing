%https://en.wikipedia.org/wiki/Support_vector_machine
% https://www.mathworks.com/help/stats/support-vector-machines-for-binary-classification.html?requestedDomain=true
% load data and remove stop words
textparsing;
remove_stop;

% training set
train_class = data_train;
train_label = (labels_train == 7);
test_class = data_test;
test_label = (labels_test == 7);

% % 5-fold cross validation
k = 5;
cv = cvpartition(train_binary_label,'KFold',k);
% set different choices of C in range 2^-9 to 2^10
% % C+ and C- by C+ = n * C/n+
C = 2.^[-9:10];
CV_CCR = zeros(k,length(C));
% 
% % for part d
CV_precision = zeros(k,length(C));
CV_recall = zeros(k,length(C));
CV_Fscore = zeros(k,length(C));
% SVM
for i = 1:k % for each fold
    train_set = train_class(cv.training(i),:);
    train_set_label = train_label(cv.training(i),:);
    test_set = train_class(cv.test(i),:);
    test_set_label = train_label(cv.test(i),:);
    for c = 1:length(C) % for each C value
        i
        c
        SVMStruct_linear = svmtrain(train_set,train_set_label,'autoscale','false','boxconstraint',C(c),'kernel_function','linear','kernelcachelimit',200000);
        Group = svmclassify(SVMStruct_linear,test_set);
        CV_CCR(i,c)= sum(Group == test_set_label)/length(Group);
        %part d
        c_matrix = confusionmat(Group,test_set_label);
        %precision = TP/^n+ = P(Y=1|h(x)=1)
        CV_precision(i,c) = c_matrix(2,2)/(c_matrix(2,2) + c_matrix(2,1));
        %recall = TP/n+ = P(h(x) = 1 | Y = 1)
        CV_recall(i,c) = c_matrix(2,2) / (c_matrix(2,2) + c_matrix(1,2));
        %recall F-score = 2PR/(P+R)
        CV_Fscore(i,c) = 2 * CV_precision(i,c) * CV_recall(i,c) / (CV_precision(i,c) + CV_recall(i,c));
        
    end
end

% summary_CV_CCR = sum(CV_CCR,2)./length(C);
summary_C_CV_CCR = sum(CV_CCR)./k;
% Plot of the CV-CCR as gunction of C
figure
hold on
plot(log2(C),summary_C_CV_CCR,'-o')
xlabel("C value in log2 scale")
ylabel("Average CV-CCR")
title("CV-CCR as function of C")
hold off

% Report the best C 
[~,index] = max(summary_C_CV_CCR,[],2);
best_C = C(index);
clear index
% % 
% % % use C to train and test on test dataset, report test CCR
SVMStruct_linear = svmtrain(full(train_class),train_label,'autoscale','false','boxconstraint',best_C,'kernel_function','linear','kernelcachelimit',200000);
Group = svmclassify(SVMStruct_linear,full(test_class));
test_CCR= sum(Group == test_label)/length(test_label);
clear SVMStruct_linear
c_matrix = confusionmat(test_label,Group);


summary_CV_precision = sum(CV_precision)./k;
summary_CV_recall = sum(CV_recall)./k;
summary_CV_Fscore = sum(CV_Fscore)./k;

% Plot of the CV-CCR as gunction of C
figure
hold on
plot(log2(C),summary_CV_precision,'-o',log2(C),summary_CV_recall,'-+',log2(C),summary_CV_Fscore,'-*')
legend({"precision","recall","F-score"})
xlabel("C value in log2 scale")
ylabel("Performance")
title("Precision, Recall, and F-score")
hold off

Report the best C 
[~,index] = max(summary_CV_recall,[],2);
best_recall = C(index);
clear index
% 
% % use C to train and test on test dataset, report test CCR
SVMStruct_linear = svmtrain(full(train_class),train_label,'autoscale','false','boxconstraint',best_recall,'kernel_function','linear','kernelcachelimit',200000);
Group = svmclassify(SVMStruct_linear,full(test_class));
recall_test_CCR= sum(Group == test_label)/length(test_label);
clear SVMStruct_linear
recall_matrix = confusionmat(test_label,Group);

[~,index] = max(summary_CV_Fscore,[],2);
best_Fscore = C(index);
clear index
% 
% % % use C to train and test on test dataset, report test CCR
SVMStruct_linear = svmtrain(full(train_class),train_label,'autoscale','false','boxconstraint',best_Fscore,'kernel_function','linear','kernelcachelimit',200000);
Group = svmclassify(SVMStruct_linear,full(test_class));
Fscore_test_CCR= sum(Group == test_label)/length(test_label);
clear SVMStruct_linear
Fscore_matrix = confusionmat(test_label,Group);
