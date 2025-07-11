clear all
load("datos\ckwc76xr2z-2\sEMG-dataset\filtered\mat\1_filtered.mat");

columna=4;
figure(1);
plot(data(:,columna));
title("Electromiografía")
xlabel("t (s)")
ylabel("Amplitud(mV)")

n = 0:319;
fs = 500;
x = data(:,columna);
signal = x;
ventana = 200000; %segundos
v = x(10:10+ventana-1);
s= fft(v);

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
title("Sección positiva de la DTFT de una señal EMG")
xlabel("f (Hz)")
ylabel("fft(f)")

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



% Rango simulado para las 3 características
[entropy, rms] = meshgrid(0:0.1:5, 0:0.1:5);  % Ejes X (entropía) e Y (rms)

% Definir superficie (puedes adaptar esta ecuación a tus datos reales)
temblor = 0.5*sin(entropy).*cos(rms) - 0.04*entropy.^2 + 0.2*rms + 2;

% Máximos y mínimos locales
[max_val, max_idx] = max(temblor(:));
[min_val, min_idx] = min(temblor(:));
[max_row, max_col] = ind2sub(size(temblor), max_idx);
[min_row, min_col] = ind2sub(size(temblor), min_idx);

% Gráfica 3D
figure;
surf(entropy, rms, temblor, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
xlabel('Entropía');
ylabel('RMS');
zlabel('Temblor');
title('Hiperplano 3D simulado con variaciones no lineales');
colormap turbo;
colorbar;
view(45, 35);
hold on;

N = length(signal);     % Número de muestras de tu señal
t = (0:N-1)/fs; 
% Suponiendo que ya tienes estas variables:
% signal: señal EMG
% t: vector de tiempo correspondiente

% Parámetros de la ventana
window_length = round(0.3 * length(t));  % duración de ventana (~30% del total)
overlap = 0.25;                          % 25% de solape
step = round(window_length * (1 - overlap));

% Colores para cada ventana
colors = {'k', 'r', 'g', 'm', 'c', [1 0.5 0], 'b', [0.5 0 0.5]};
line_styles = {'--', '--', '--', '--', '--', '--', '--', '--'};

% Dibujar la señal
figure;
plot(t, signal, 'b'); hold on;
xlabel('Tiempo (s)');
ylabel('Amplitud (mV)');
ylim([min(signal)-0.5, max(signal)+0.5]);

% Dibujar ventanas superpuestas
start_idx = 1;
color_idx = 1;
while start_idx + window_length <= length(t)
    x0 = t(start_idx);
    x1 = t(start_idx + window_length);
    y0 = min(signal)-0.5;
    y1 = max(signal)+0.5;
    
    % Dibujar rectángulo sin relleno
    rectangle('Position', [x0, y0, x1-x0, y1-y0], ...
              'EdgeColor', colors{color_idx}, ...
              'LineStyle', line_styles{color_idx}, ...
              'LineWidth', 2);
    
    % Avanzar
    start_idx = start_idx + step;
    color_idx = mod(color_idx, length(colors)) + 1;
end

title('Segmentación de la señal en ventanas solapadas');

precision = [0.918,0.913,0.920,0.872,0.874];
tpred = [1,2,3,4,5];

figure;
plot (tpred, precision)
xlabel("Tiempo de predicción (s)")
ylabel("F1 score promedio")
ylim([0.7:1]);