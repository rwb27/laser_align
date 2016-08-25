include <C:\Users\a-amb\OneDrive - University Of Cambridge\openflexure_microscope-master\openscad\microscope_parameters.scad>;
use <utilities.scad>;

/*************************************************************
Simple butt-coupled LED-to-SMA mount for aligning microscopes
with spectrometers coupled by a fibre patch cable.

(c) Richard Bowman 2015, please share under CC-BY 3.0
You can use it for what you like, but attribution would be 
very nice.  Thanks!
*************************************************************/

SMA_r = 3.2/2+0.2; //dimensions of SMA ferrule
SMA_h = 10;
SMA_thread = 4;
spacing = 0.2; //standoff between LED and bottom
mounting_holes = [[leg_middle_w/2,leg_r-zflex_l-4,0], -1*[leg_middle_w/2,leg_r-zflex_l-4,0]];

h = spacing + SMA_h;

$fn=32;
d=0.05;

difference(){
    union(){
        cylinder(r=6,h=4); //main body
        hull() for(h=mounting_holes) translate(h) cylinder(r=10,h=3);
    }
    
    cylinder(r=1.25,h=spacing+d);
    translate([0,0,spacing]) cylinder(r=1.25,h=999);
    
    //mounting screws
    for(h=mounting_holes) translate(h) cylinder(r=2.8*1.2/2,h=999,center=true); //originally 2.8 was 3
}
