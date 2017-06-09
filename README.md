# Range of value of Perlin Noise

This respository contains a script to:

1) Find the largest value of 2D Perlin Noise.

2) Find the largest value of 3D Perlin Noise.

3) Find the largest value of the magnitude of the gradient of 2D Perlin Noise.

4) Find the largest value of the magnitude of the gradient of 3D Perlin Noise.

The key techniques used are [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) and [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation). **Gradient descent** (or ascent) just means "keep going up the hill until you can't go up the hill anymore". Perlin noise is made of many hills though, so hill-climbing is done multiple times, starting at random points. However, hill-climbing works best if the "up" direction of the hill is known. This is determined using the gradient (derivative) of Perlin Noise. This can be worked out analytically, but it's cool to use **automatic differentiation** which calculates the derivative of *any* function with respect to its arguments. Even if it contains branches, loops and function calls!

# Purpose

If you're here, I'm assuming you came from Google looking for these numbers.

If not, the context is that Perlin Noise is a form of [coherent noise](http://libnoise.sourceforge.net/glossary/index.html#coherentnoise) widely used in computer graphics to generate good-looking random patterns. Any implementation won't necessarily generate noise in the range of [-1, 1], but users of random number generators typically expect to be able to specify the range. To have an exact range, you need to know the range of values that the function can produce and apply the right scaling factor.

# Usage

This script requires the packages `scipy`, `numpy`, `matplotlib` and `autograd`.

To find the maximum values of Perlin Noise, use `./findrange.py`.

To plot some 2D Perlin Noise functions and the gradient magnitude (for debugging), use `./plotperlin.py`.

![3x3 Perlin Noise](/images/perlin1.png)

Sample 2D Perlin Noise and gradient magnitude (3x3 grid)

![Perlin Noise Maximum](/images/perlin2.png)

Unique configuration of 2D Perlin Noise vectors that maximizes the function.

![Perlin Noise Gradient Maximum](/images/perlin3.png)

One possible configuration of 2D Perlin Noise vectors that maximizes the gradient magnitude.

# Technical Details

In 2D, the value of Perlin Noise at any point depends on the coordinate (x, y) and the vectors at each corner of the corresponding grid cell. Generally, the vectors are of unit length, so they can be represented using an angle. All cells are statistically identical, so we only need to find the largest value of Perlin Noise within one cell. That means we optimize a function of 6 variables (angle1, angle2, angle3, angle4, x, y).

In 3D, the mechanism is the same, but now the cell is a cube with 8 corners and each angle needs to be represented with 2 angles. That means we optimize a function of 19 variables (x, y, z) and 2 * 8 angles.

To complicate matters, Perlin Noise can refer to a few different implementations.

First, Perlin Noise is often confused with [Value Noise](https://en.wikipedia.org/wiki/Value_noise). Determining the range of values of Value Noise is trivial and doesn't require optimization techniques.

Second, Ken Perlin published an [Improved Perlin Noise](http://http.developer.nvidia.com/GPUGems/gpugems_ch05.html) where he made a few tweaks. He changed the original interpolation function 3t^2 - 2t^3 to 6t^5 - 15t^4 + 10t^3 which has better continuity properties. This doesn't affect the range of values of Perlin Noise but it does affect the value and location of the maximum in the gradient magnitude.

Third, these are many different ways to select the random vectors at the grid cell corners. In Improved Perlin Noise, instead of selecting any random vector, one of 12 vectors pointing to the edges of a cube are used instead. This script optimizes using a continuous range of angles since it is easier - however, the range of value of an implementation of Perlin Noise using a restricted set of vectors will never be larger. Furthermore, the script in this repository assumes the vectors are of unit length. If they not, the range of value should be scaled according to the maximum vector length. Note that the vectors in Improved Perlin Noise are not unit length.

# Results

| Function   | Range of values   |
| ---------- | ----------------- |
| 2D Perlin  | [-0.707, +0.707]  |
| 3D Perlin  | [-0.866, +0.866]  |
| 2D Perlin gradient magnitude (3rd order) | [0, +1.807] |
| 2D Perlin gradient magnitude (5th order) | [0, +2.072] |
| 3D Perlin gradient magnitude (3rd order) | [0, +2.182] |
| 3D Perlin gradient magnitude (5th order) | [0, +2.793] |

(Those numbers are rounded, you might want to increase the last digit in your application.)

Note that Perlin Noise is symmetric around 0 so the maximum value corresponds is the same as the minimum value.

For more details, see [results](results.md).
