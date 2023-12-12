clear;
clc;
close;
deap = load('C:\Users\praka\Desktop\courses\11777\DEAP_procdata\s01.mat');
amigos = load('C:\Users\praka\Desktop\courses\11777\AMIGOS_procdata\Data_Preprocessed_P01\Data_Preprocessed_P01.mat');
figure(2)
hold on 
x1 = linspace(0,8064/128,8064);
x2 = linspace(0,size(amigos.joined_data{1}(:,2),1)/128,size(amigos.joined_data{1}(:,2),1));
plot(x2, amigos.joined_data{1}(:,2))
plot(x1,reshape(deap.data(1,4,:),1,8064))
xlim([0 60])
xlabel('Time (s)')
ylabel("EEG ")
legend('AMIGOS F7 EEG','DEAP F7 EEG')
