
def cropper(x,y,w,h):
    """create a function to crop the specified region out of an image"""
    def crop(image):
        """Crop an image"""
        return image[x:x+w, y:y+h, ...]
    return crop

crop_middle = cropper(100,100,50,50)

cropped_image = crop_middle(image)

cropped_image = cropper(100,100,50,50)(image)