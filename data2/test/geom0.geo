lc1 = 1.750; 
lc2 = 2.000; 

// border points of rectangle 
Point(1) = {-1.0, -1.0, 0.0, lc1}; 
Point(2) = { 1.0, -1.0, 0.0, lc1}; 
Point(3) = { 1.0,  1.0, 0.0, lc1}; 
Point(4) = {-1.0,  1.0, 0.0, lc1}; 

// boundary lines of rectangle 
Line(1) = {1, 2}; 
Line(2) = {3, 2}; 
Line(3) = {3, 4}; 
Line(4) = {4, 1}; 

Curve Loop(1) = {4, 1, -2, 3}; 
Plane Surface(1) = {1}; 

