def draw_triangle(height, level=1):
    if level > height:
        return  # base case
    else:
        print(" " * (height - level) + "*" * level + " " * (height - level))
        draw_triangle(height, level + 1)


draw_triangle(3)
draw_triangle(5)
