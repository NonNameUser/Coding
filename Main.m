clear;clc;
close all;
% GaitUI System
data = load('user_A_5.mat');   
csi_trace=data.csi_trace;
channel=30; %Number of channels
len=length(csi_trace);  %Signal length
data=zeros(len,channel*3);  %Raw CSI data
time=zeros(1,len);  %Time stamp
c=3*10^8;       %Speed of light
rx_acnt=3;      %Antenna count for the receiver

%Extracting CSI data from the file
for i=1:len 
    csi_entry = csi_trace{i};
	csia=get_scaled_csi(csi_entry);  
    data(i,1:30)=csia(1,1,1:30);    %Antenna 1 raw CSI signal
    data(i,31:60)=csia(1,2,1:30);   %Antenna 2 raw CSI signal
    data(i,61:90)=csia(1,3,1:30);   %Antenna 3 raw CSI signal
    time(i)=csi_entry.timestamp_low;
end
csi_stream=data;

%CSI Ratio Calculation
idx=3;
csi_data_ref = repmat(csi_stream(:,(idx-1)*30+1:idx*30), 1, rx_acnt);
abs_csi_data_ref=abs(csi_data_ref);
Angle_csi_data_ref=angle(csi_data_ref);
csi_data_ref=(abs_csi_data_ref*1000).*exp(1j*Angle_csi_data_ref);
csi_ratio=csi_stream./csi_data_ref;
csi_ratio=[csi_ratio(:,1:30*(idx - 1)) csi_ratio(:,30*idx+1:90)];

%Static component elimination
diff_csi_ratio=diff(csi_ratio,1);
time=(time-time(1))/10^6;   %time(s)
conj_data=diff_csi_ratio.';
[~,r,~] = find(isnan(conj_data)|isinf(conj_data));
conj_data(:,r)=[];
time(r)=[];

conj_data=lowpass(conj_data,40,500);
window=0.15*500;        %window length
num=50;  
v=linspace(-8,8,num);   %Velocity
f=5220*10^6;            %center frequency
i=1;
ind=1;
M=1;                    %Number of sources
len=size(conj_data,2);
time_picture=[];

%Velocity profile generation
while i<len-window
    P(:,ind)=zeros(num,1);
    ch=1;
    X1=conj_data(1:30,i:i+window-1).';
    for row=1:size(X1,2)
        X1(:,row)=X1(:,row)/size(X1,1);
    end
    T=time(i:i+window-1);
    t=time(i+1:i+window);
    time_picture=[time_picture,time(i)];
    N=length(T);
    Rxx=X1*X1'/window;
    %Eigenvalue decomposition
    [Gn,D]=eig(Rxx);           
    EVA=diag(D)';                      
    [EVA,I]=sort(EVA);                
    Gn=Gn(:,I);                
    q = EVA(1);
    SUM = 0;
    for k = N-M+1:N
        u = Gn(:,k);
        SUM = SUM+(EVA(k)*u*u')/(q-EVA(k))^2;
    end
    U = q*SUM;
    G=Gn(:,1:N-M)*Gn(:,1:N-M)';   % Constructing noise subspaces
    %%%%%%%Traverse each velocity and calculate the velocity spectrum%%%%%%
    for iV= 1:num
        a=exp(-1i*2*pi*f.*v(iV).*T/c).'-exp(-1i*2*pi*f.*v(iV).*t/c).';  
        Pmusic(iV)=(a'*U*a)/(a'*G*a);
    end
    Pmusic=abs(Pmusic);
    P(:,ind)=P(:,ind)+Pmusic'; 
    i=i+8;
    ind=ind+1;
end
time_index=i;
%%
new_P2=P;
%Normalization
for i=1:size(new_P2,2)
    new_P2(:,i)=  new_P2(:,i)/sum(new_P2(:,i));
    if sum(P(12:25,i))<0.05
        new_P2(:,i)=  new_P2(:,i)*0.002;
    end
end
set(gcf,'position',[200 150 800 400])
x=linspace(0,time(time_index),size(new_P2,2));
y=v/2;
[X,Y]=meshgrid(x,y);
mesh(X,Y,new_P2)
shading interp
view(0,90)
xlabel('\fontname{Times New Roman}Time(s)','fontsize',14)
ylabel('\fontname{Times New Roman}Velocity(m/s)','fontsize',14)
colormap(jet)
xlim([0,x(end)])
set(gca,'FontSize',20);
set(gca,'FontWeight','bold')
set(gca, 'LooseInset', [0,0,0,0]);
%%
%Cumulative signal energy
signal_energy=zeros(1,size(P,2));
for i=1:size(P,2)
    signal_energy(i)=sum(P(:,i));
end
Eta=zeros(1,size(P,2));
K=1e+17;
for i=1:size(P,2)
    Eta(i)=mean(signal_energy(1:i))+K*std(signal_energy(1:i));
end
order = 1;
framelen = 61;
Eta = smooth(sgolayfilt(Eta,order,framelen),50);
T=time_picture;
set(gcf,'position',[200 150 800 400])
x=linspace(0,time(4000),length(Eta));
h=plot(x,Eta);
hold on
xlabel('\fontname{Times New Roman}Time(s)','FontSize',14);
ylabel('\fontname{Times New Roman}Energy','FontSize',14);
set(h,'Linewidth',3)
xlim([0,8])
ylim([0,15])
index_eta=find((Eta(20:end))>=1.15)+19;
point=plot(T(index_eta(1)),Eta(index_eta(1)),'.r','MarkerSize',50);
x_num=linspace(0,10,1000);
y_num=ones(1,1000)*Eta(index_eta(1));
h_y=plot(x_num,y_num,'-r');
set(h_y,'Linewidth',3)
x_num1=ones(1,1000)*T(index_eta(1));
y_num1=linspace(0,80,1000);
h_x=plot(x_num1,y_num1,'--r');
set(h_x,'Linewidth',3)
set(gca,'FontSize',18);
set(gca,'FontWeight','bold')
legend([h point],{'\fontname{Times New Roman}Cumulative energy','\fontname{Times New Roman}Point 1'},'FontSize',14)
set(gca, 'LooseInset', [0,0,0,0]);
%%
%Cumulative phase variation
CSII_quotient_data=data(:,1)./data(:,31);
order = 5;
framelen = 21;
filt_CSII_quotient_data = sgolayfilt(CSII_quotient_data,order,framelen);
diff_filt_CSII_quotient_data=diff(filt_CSII_quotient_data);
Diff_Angle=smooth(unwrap(angle(diff_filt_CSII_quotient_data)),50);
[~,index]=max(Diff_Angle(1:4000));
index_point2=floor(index/8)+(mod(index,8)>1);
set(gcf,'position',[200 150 800 400])
h=plot(T,Diff_Angle(1:8:length(T)*8));
xlabel('\fontname{Times New Roman}Time(s)','FontSize',14);
ylabel('\fontname{Times New Roman}Phase(rad)','FontSize',14);
set(h,'Linewidth',3)
hold on
xlim([0 8])
ylim([-200 2000])
x_num2=linspace(0,10,1000);
y_num2=ones(1,1000)*Diff_Angle(index);
h1=plot(x_num2,y_num2,'-r');
set(h1,'Linewidth',3)
x_num3=ones(1,1000)*T(index_point2);
y_num3=linspace(-200,2100,1000);
h3=plot(x_num3,y_num3,'r--');
set(h3,'Linewidth',3)
point=plot(T(index_point2),Diff_Angle(index),'.r','MarkerSize',50);
legend([h point],{'\fontname{Times New Roman}Cumulative phase','\fontname{Times New Roman}Point 2'},'FontSize',18,'Location','northwest')
set(gca,'FontSize',18);
set(gca,'FontWeight','bold')
set(gca, 'LooseInset', [0,0,0,0]);
ax = gca;
ax.YAxis.Exponent = 3;
%%
set(gcf,'position',[200 150 800 400])
A=new_P2;
number_picture=size(new_P2,1);
x=T;
y=v/2;
[X,Y]=meshgrid(x,y);
surf(X,Y,A)
shading interp
view(0,90)
xlabel('\fontname{Times New Roman}Time(s)','fontsize',14)
ylabel('\fontname{Times New Roman}Velocity(m/s)','fontsize',14)
ylim([y(1) y(end)])
xlim([x(1) x(end)])
hold on
x_num4=ones(1,1000)*T(index_point2);
y_num4=linspace(-200,1500,1000);
z_num4=ones(1,1000)*1;
h4=plot3(x_num4,y_num4,z_num4,'y--');
set(h4,'Linewidth',2)
x_num5=ones(1,1000)*time(index_eta(1)*8);
y_num5=linspace(-200,1600,1000);
z_num5=ones(1,1000)*1;
h5=plot3(x_num5,y_num5,z_num5,'--r');
set(h5,'Linewidth',2)
set(get(gca,'XLabel'),'FontSize',14);
set(get(gca,'YLabel'),'FontSize',14);
set(gca,'FontSize',18);
set(gca,'FontWeight','bold')
colormap(jet)
set(gca, 'LooseInset', [0,0,0,0]);
%%
set(gcf,'position',[200 150 800 400])
B=new_P2;
plot_B=B;
x=T(index_eta(1):index_point2);
y=linspace(-4,0,num/2);
[X,Y]=meshgrid(x,y);
mesh(X,Y,plot_B(1:num/2,index_eta(1):index_point2))

view(0,90)
xlabel('\fontname{Times New Roman}Time(s)','fontsize',14)
ylabel('\fontname{Times New Roman}Velocity(m/s)','fontsize',14)
ylim([y(1) y(end)])
xlim([x(1) x(end)])
set(get(gca,'XLabel'),'FontSize',14);
set(get(gca,'YLabel'),'FontSize',14);
set(gca,'FontSize',14);
colormap(jet)
set(gca,'FontSize',18);
set(gca,'FontWeight','bold')
set(gca, 'LooseInset', [0,0,0,0]);
%%
%Leg Velocity Estimation
C=B;
C(1:5,:)=C(1:5,:)*0;
C(1:13,1:100)=C(1:13,1:100)*0;
velocity=zeros(1,size(C,2));
V_B=zeros(1,size(C,2));
for i=1:size(C,2)
    for j=1:size(C,1) 
        if sum(C(1:j,i))>=0.05*(sum(C(1:25,i)))
            velocity(i)=j;
            V_B(i)=v(j);
            break;
        end
    end
end
velocity=(velocity-num/2)/num*8;
time_B=T(index_eta(1):index_point2);
h=plot(time_B,velocity(index_eta(1):index_point2),'b-');
set(h,'lineWidth',2)
xlabel('')
xlim([x(1),x(end)])
ylim([-4,0])
set(gcf,'position',[200 150 600 310])
h_velocity=plot(time_B,velocity(index_eta(1):index_point2),'b-');
set(h_velocity,'Linewidth',3)
xlim([time_B(1),time_B(end)])
ylim([-4,0])
xlabel('\fontname{Times New Roman}Time(s)','FontSize',14)
ylabel('\fontname{Times New Roman}Velocity(m/s)','FontSize',14)
%%
%Gait Duration Discovery
set(gcf,'position',[200 150 600 310])
h_velocity=plot(time_B,velocity(index_eta(1):index_point2),'b-');
set(h_velocity,'Linewidth',3)
hold on
smooth_velocity=velocity;
D=hampel(smooth_velocity,7,4);
D=smooth(D,15);
D=medfilt1(D,9);
order = 1;
framelen = 7;
D= sgolayfilt(D,order,framelen);
h_D=plot(time_B,D(index_eta(1):index_point2),'r-.');
set(h_D,'Linewidth',3)
xlim([time_B(1),time_B(end)])
ylim([-4,0])
xlabel('\fontname{Times New Roman}Time(s)','FontSize',14)
ylabel('\fontname{Times New Roman}Velocity(m/s)','FontSize',14)
legend('\fontname{Times New Roman}Original','\fontname{Times New Roman}Smoothed','FontSize',12,'Location','SouthEast')
[value_maxValue,points_maxValue]=findpeaks(D(index_eta(1):index_point2));
[value_minValue,points_minValue]=findpeaks(-D(index_eta(1):index_point2));
Steading_Walking_Begin=1;
for i=1:length(points_minValue)
    if value_minValue(i)>1
        for j=1:length(points_maxValue)
            if(points_maxValue(j)>points_minValue(i))
                Steading_Walking_Begin=points_maxValue(j);
                break;
            end
        end
        break;
    end
end
index=find(D(index_eta(1):index_point2)==value_maxValue(end-2));
rectangle('position',[time_B(index(1)),-4,time_B(end)-time_B(index(1))-0.01,4],'LineStyle','--','LineWidth',3,'EdgeColor','m')
text((time_B(end)+time_B(index(1)))/2-0.25,-0.5,'\fontname{Times New Roman}FDZ','FontSize',20,'Color','m','FontWeight','bold')
rectangle('position',[time_B(1),-4,time_B(Steading_Walking_Begin)-time_B(1),4],'LineStyle','--','LineWidth',3,'EdgeColor',[0 0.39216 0])
text(time_B(1)+0.015,-0.5,'\fontname{Times New Roman}NSW','FontSize',20,'Color',[0 0.39216 0],'FontWeight','bold')
rectangle('position',[time_B(Steading_Walking_Begin)+0.04,-4,time_B(index(1))-time_B(Steading_Walking_Begin)-0.08,4],'LineStyle','--','LineWidth',3,'EdgeColor',[1 0.54902 0])
text(2,-0.5,'\fontname{Times New Roman}Steady walking','FontSize',20,'Color',[1 0.54902 0],'FontWeight','bold')
set(gca,'FontSize',14);
set(gca,'FontWeight','bold')
set(gca, 'LooseInset', [0,0,0.01,0]);
%%
K=C(1:num/2,index_eta(1):index_point2);
last_velocity=D(index_eta(1):index_point2);
point1=Steading_Walking_Begin-10;
point2=index(1)+20;
subplot(2,1,1)
plot(time_B(point1:point2),last_velocity(point1:point2))
xlabel('时间')
ylabel('速度')
subplot(2,1,2)
x=time_B(point1:point2); 
y=linspace(-4,0,num/2);
[X,Y]=meshgrid(x,y);
mesh(X,Y,K(:,point1:point2));
view(0,90)
xlabel('时间')
ylabel('速度')
hold on
plot3(time_B(point1:point2),last_velocity(point1:point2),ones(1,length(x))*1,'r--','LineWidth',3); 
colormap(jet)
%%
%Velocity Profile Segmentation
set(gcf,'position',[200 150 600 310])
[~,L1] = findpeaks(last_velocity(point1:point2));
for i=1:length(L1)
    if last_velocity(point1-1+L1(i))<-3.5
        L1(i:end)=[];
        break;
    end
end

new_C=plot_B(:,index_eta(1)+point1-1:index_eta(1)+point2+10);
new_B=new_C;
if L1(1)<=3
    step=L1(1)-1;
else
    step=3;
end
time_sample=zeros(1,length(L1));

num_time_sample=0;
if length(L1)>=1
    for i=1:length(L1)
        if L1(i)+step-1<=size(new_C,2)
            [~,temp]=find(new_C(:,L1(i)-step:L1(i)+step)==max(max(new_C(:,L1(i)-step:L1(i)+step))));
        time_sample(i)=temp(1);
        time_sample(i)=time_sample(i)+L1(i)-step;
        num_time_sample=num_time_sample+1;
        else
            break;
        end
    end    
end
sigma = 1;
gausFilter = fspecial('gaussian', [3,3], sigma);
new_C= imfilter(new_C, gausFilter, 'replicate');

for i=1:size(new_B,2)
    for j=1:size(new_B,1)
        if new_B(j,i)>4e-4
            new_B(j,i)=4e-4;
        end
    end
end
x=time_B(point1:point1-1+L1(end)+5);  
y=linspace(-4,0,num/2);
[X,Y]=meshgrid(x,y);
surf(X,Y,new_B(1:num/2,1:L1(end)+5));
shading interp
view(0,90)

xlim([x(1),x(end)]) 
hold on
for k=1:num_time_sample
        if time_sample(k)>length(x)
            time_sample(k)=length(x);
        end
    h=plot3(ones(1,length(y))*(x(time_sample(k))),y,ones(1,length(y))*10,'r--');%plot3(ones(1,length(y))*x(time_sample(k)),y,ones(1,length(y))*10,'r-');
    set(h,'lineWidth',3)
    hold on
end

colormap(jet)
xlabel('\fontname{Times New Roman}Time(s)','FontSize',16)
ylabel('\fontname{Times New Roman}Velocity(m/s)','FontSize',16)
axis off
set(gca,'FontSize',14);
set(gca,'FontWeight','bold')
set(gca, 'LooseInset', [0,0,0,0]);
