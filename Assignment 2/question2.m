% Focal length in pixels
F=961;
 
% Eight corresponding points of two images
points1 = [[129 322]; [544 1127]; [225 133]; [335 168]; [380 341]; [210 738]; [210 748]; [224 738];];
points2 = [[304 318]; [725 1106]; [390 114]; [499 137]; [544 309]; [385 708]; [386 717]; [399 708];];

% Center coordinates of image
x_center = 960/2;
y_center = 1280/2;

% Obtaining N-vectors
image1_points = [points1 ones(8,1)*F];
image1_points(: ,1) = image1_points (: ,1) - X0 ;
image1_points (: ,2) = image1_points (: ,2) - Y0 ;
image1_norm = vecnorm (image1_points , 2 , 2) ;
image1_points = image1_points ./ image1_norm ;

image2_points = [points2 ones(8,1)*F];
image2_points (: ,1) = image2_points (: ,1) - X0 ;
image2_points (: ,2) = image2_points (: ,2) - Y0 ;
image2_norm = vecnorm (image2_points , 2 , 2) ;
image2_points = image2_points ./ image2_norm ;

% Creating W matrix
weight_matrix = zeros (3 ,3) ;
for i = 1:8
temp = transpose ( image1_points (i ,:) ) * image2_points (i ,:) ;
weight_matrix = weight_matrix + temp ;
end

% SVD approach
[U , S , V ] = svd( W ) ;
rotational_matrix = U * transpose ( V ) ;

angleInRadians = acos((trace(rotational_matrix)-1)/2);
angleInDeg = rad2deg(angleInRadians);
% angleInDeg = 10.2375 degrees;

l = transpose ([ rotational_matrix(3 ,2)-rotational_matrix(2 ,3) rotational_matrix(1 ,3)-rotational_matrix(3 ,1) rotational_matrix(2 ,1)-rotational_matrix(1 ,2)]) ;
l = l ./ vecnorm(l);
% N vector of rotational axis, [-0.1640 -0.9831 0.0814 ]