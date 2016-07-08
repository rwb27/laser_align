

def thing(ting):
    return ting**2


def thing2(yeah, n='banana'):
    return yeah**3, n

func_list = [[thing, 3],
        [thing2, 4, 'IMAGE_ARR']]


for function in func_list:
    for index, item in enumerate(function):
        if item == 'IMAGE_ARR':
            # Replace 'IMAGE_ARR' in each list by the image
            # array.
            function[index] = 'yeah'
    print function

    print function[0](*function[1:])
