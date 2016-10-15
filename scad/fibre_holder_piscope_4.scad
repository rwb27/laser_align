include <C:\Users\Abhishek\OneDrive - University Of Cambridge\Resources\Programming\openflexure_microscope\openscad\microscope_parameters.scad>;
use <utilities.scad>;

/*************************************************************
Simple butt-coupled LED-to-SMA mount for aligning microscopes
with spectrometers coupled by a fibre patch cable.

(c) Richard Bowman 2015, please share under CC-BY 3.0
You can use it for what you like, but attribution would be 
very nice.  Thanks!
*************************************************************/

SMA_h = 10;
spacing = 0.2; //standoff between LED and bottom
mounting_holes = [[leg_middle_w/2,leg_r-zflex_l-4,0], -1*[leg_middle_w/2,leg_r-zflex_l-4,0]];

h = spacing + SMA_h;

//thing = translate([-25, 25]){cube([50,50,3])};
difference(){
    translate([0, 0, 1.5]){rotate([0, 0, 45]){cube([52, 52, 3], center=true);}}
    
    cylinder(r=7,h=3);
    cylinder(r=13.5, h=2.4);
    translate([11, 11, 0]){cylinder(r=1.4, h=3);}
    rotate([0, 0, 90]){translate([11, 11, 0]){cylinder(r=1.3, h=3);}}
    rotate([0, 0, 180]){translate([11, 11, 0]){cylinder(r=1.3, h=3);}}
    rotate([0, 0, 270]){translate([11, 11, 0]){cylinder(r=1.3, h=3);}}

    
    //mounting screws
    rotate([0, 0, 14]){
    for(h=mounting_holes) translate(h) cylinder(r=3*1.2/2,h=999,center=true);
    rotate([0, 0, 90]){for(h=mounting_holes) translate(h) cylinder(r=3*1.2/2,h=999,center=true);}}
}
