# Integrals Intuitions

If $F’(x)$ is the derivative of $F(x)$

then the integral of $F’(x)$ is $F(x) + c$.

$F’(x)$ is the integrand and $F(x) + c$ is the integral. 

An interesting relationship between $F’(x)$ and $F(x)$ is that each point of $F’(x)$ is how much $F(x)$ is changing at that point. This is true for all $g'(x)$ and $g(x)$, by definition of derivative, not integral.

So that means, if you start at some value of $F(x)$, call it $c$, you can add up all those little changes from that point up to another point.

So add up all the $F’(x + \text{\color{green}in between however much})$ to $c$ and you get $F(x + \text{\color{green}however much})$.

So $F(x + \text{however much}) - F(x)$ is the sum of all those little changes between $x$ and $x + \text{\color{green}however much}$.

In other words, it’s the sum of all the lines under $F’(\cdot)$ from that start to end point.

Those lines next to each other geometrically (like thin rectangles) are just the area under the curve for $F’(\cdot)$ in that range.

Therefore, the area under the curve of the integrand $F’(x)$ for a specific range is the difference between the two integrals: $F(x + \text{\color{green}however much}) - F(x)$.

And when a range is specified over which to do that subtraction of two integrals, that’s called a definite integral. Otherwise, those individual integrals are referred to as indefinite integrals.

And that’s all.

## Definitions

### Indefinite Integral

An indefinite integral is just an inverse derivative. If something is the derivative of something else, then something else is the indefinite integral of something. 

It’s a function because something can hypothetically be the derivative of multiple hypothetical something elses, varying in difference by a constant c, since taking the derivative eliminates that constant anyway. 

> It looks like this:
> 
> $$F(x) = \int f(x) dx,$$
>
> and the only requirement in this definition is that $F'(x) = f(x)$.

### Definite Integral

A definite integral is the difference of two something elses at two different points, an upper point and a lower point. 

It’s a value rather than a function because the multiple something elses that might be the integral differ only by the same amount c, and that amount gets eliminated by the subtraction, leaving only a definite value. 

> It looks like this:
> 
> $$F(b) - F(a) = \int_{a}^{b} f(x) dx.$$
> 
> The meaning of $dx$ here is the little widths of those rectangles that get added up in the area under the curve (with $f(x)$ as the heights of those respective rectangles). $a$ and $b$ are the two points.
