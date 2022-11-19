
-- | In Futhark we take a slice out of an array with numpy slice syntax, but
--   there doesn't seem to be any way to ascribe a static shape to this slice
--   even if we do know it statically.
--
--   If we leave the return type dynamic like this then we will get a
--   unification error when we try to use it.
--
def slice4
      [nAi][nAc][nAh][nAw]
      (arrA: [nAi][nAc][nAh][nAw]f32)
      (start: (i64, i64, i64, i64))
      (slice: (i64, i64, i64, i64))
    : ([][][][]f32) =
  arrA[
    start.0 : start.0 + slice.0,
    start.1 : start.1 + slice.1,
    start.2 : start.2 + slice.2,
    start.3 : start.3 + slice.3]

-- Dot product.
def dot [n]
      (arrA: [n]f32)
      (arrB: [n]f32)
    : f32 =
  reduce (\x y -> x + y) 0 ((map (\(x,y) -> x * y) (zip arrA arrB)))

-- Enumerate Nx1xHxW indices.
def indices_n1hw
    (n: i64)(h: i64)(w: i64)
    : [n][1][h][w](i64, i64, i64, i64) =
  (map (\i0 ->
    map (\_ ->
      map (\i2 ->
        map (\i3 ->
          (i0, 0, i2, i3))
          (0..<w))
        (0..<h))
      [1])
    (0..<n))

-- Enumerate Nx1x1x1 indices.
def indices_n111
    (n: i64)
    : [n][1][1][1](i64, i64, i64, i64) =
  (map (\i ->
    map (\_ ->
      map (\_ ->
        map (\_ ->
          (i, 0, 0, 0))
          [1])
        [1])
      [1])
    (0..<n))

-- | Direct unpadded 2D convolution with unit stride and no dilation, over the
--   spatial dimensions of a rank 4 array.
--
--   A is the input array of shape (nAi, nAc, nAh, nAw), where the dimensions
--   are (images, channels, height, width) respectively.
--
--   K is the array of convolution filters of shape (nBc, nAc, nKh, nKw).
--
--   B is the output array of shape (nAi, nBc, nBh, nBw), where the two
--   outermost dimensions are determiend by those of A and K, and the spatial
--   dimensions are the same as A.
--
def conv2d
      [nAi][nAc][nAh][nAw]
      [nBc][nKh][nKw]
      [nBh][nBw]
      (arrA: [nAi][nAc][nAh][nAw]f32)
      (arrK: [nBc][nAc][nKh][nKw]f32)
    : ([nAi][nBc][nBh][nBw]f32) =
  let shK1 = (1, nAc, nKh, nKw)
  -- The following doesn't work. Not cool with slices of unknown static sizes.
  -- But we can't use slice syntax [:] to produce slices of static size.
  -- What to do?
  let psA  = map (\i -> slice4 arrA i shK1) (indices_n1hw nAi nAh nAw)
  let psK  = map (\i -> slice4 arrK i shK1) (indices_n111 [nBc])
  in map (\pA -> map (\pK -> dot pA pK) psK) psA
