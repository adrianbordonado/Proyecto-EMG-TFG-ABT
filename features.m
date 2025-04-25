clear all
load("datos\ckwc76xr2z-2\sEMG-dataset\filtered\mat\1_filtered.mat");

columna=4;
figure(1);
plot(data(:,columna));

n = 0:319;
fs = 500;
x = data(:,columna);
ventana = 200000; %segundos
v = x(10:10+ventana-1);
s= fft(v);

%VENTANA DE LA SEÑAL

figure(1)
plot(v)

%ENTROPÍA

ent = pentropy(v,fs);
figure(2)
plot(ent)


%FRECUENCIA MEDIA

figure(3)
plot(fs/ventana*(-ventana/2:ventana/2-1),abs(fftshift(s)),"LineWidth",2)
title("fft Spectrum in the Positive and Negative Frequencies")
xlabel("f (Hz)")
ylabel("|fft(X)|")

P2 = abs(s/ventana);
P1 = P2(1:ventana/2+1);
P1(2:end-1) = 2*P1(2:end-1);

figure(4)
f = fs/ventana*(0:(ventana/2));
plot(f,P1,"LineWidth",2) 
title("Single-Sided Amplitude Spectrum of X(t)")
xlabel("f (Hz)")
ylabel("|P1(f)|")

mean(P1);

%MAV

mean(abs(v));

%FRECUENCIA PREDOMINANTE

[psd_welch,w1] = pwelch(v);
figure(6)
plot(psd_welch)

%POTENCIA MEDIA

mean(psd_welch);

%RMS

rms= sqrt(sum(v.^2)/length(v));
ams= (sum(abs(v))/length(v));

%AGUDEZA

k= kurtosis(v)

%ASIMETRÍA

sk= skewness(v)