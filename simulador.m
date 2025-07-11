clear all
load('datos/svm_data_grid.mat');

% Asegúrate que las variables tienen las formas correctas
decision = decision(:); % vector columna
y = y(:);

% --- Filtro IQR para limpiar outliers ---
Q1 = quantile(X_pca, 0.25);
Q3 = quantile(X_pca, 0.75);
IQR = Q3 - Q1;

lower_bound = Q1 - 1.5 * IQR;
upper_bound = Q3 + 1.5 * IQR;

mask_iqr = all(bsxfun(@ge, X_pca, lower_bound) & bsxfun(@le, X_pca, upper_bound), 2);

X_filtered = X_pca(mask_iqr,:);
y_filtered = y(mask_iqr);
decision_filtered = decision(mask_iqr);

fprintf('Datos iniciales: %d\n', size(X_pca,1));
fprintf('Tras filtro IQR: %d\n', size(X_filtered,1));

figure; hold on; grid on;

scatter3(X_filtered(y_filtered==1,1), X_filtered(y_filtered==1,2), X_filtered(y_filtered==1,3), 36, 'r', 'filled');
scatter3(X_filtered(y_filtered==-1,1), X_filtered(y_filtered==-1,2), X_filtered(y_filtered==-1,3), 36, 'b', 'filled');

[xq, yq, zq] = meshgrid(...
    linspace(min(X_filtered(:,1)), max(X_filtered(:,1)), 30), ...
    linspace(min(X_filtered(:,2)), max(X_filtered(:,2)), 30), ...
    linspace(min(X_filtered(:,3)), max(X_filtered(:,3)), 30));

F = scatteredInterpolant(X_filtered(:,1), X_filtered(:,2), X_filtered(:,3), decision_filtered);
vq = F(xq, yq, zq);

p = patch(isosurface(xq, yq, zq, vq, 0));
set(p, 'FaceColor', 'k', 'EdgeColor', 'none', 'FaceAlpha', 0.3);

xlabel('Entropía'); ylabel('RMS (mV)'); zlabel('Temblor');
title('Frontera de decisión SVM kernel RBF en 3D (PCA)');
view(3);
camlight; lighting gouraud;
