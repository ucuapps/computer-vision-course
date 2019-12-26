def line_exp_perp_2d ( p1, p2, p3 ):

#*****************************************************************************80
#
## LINE_EXP_PERP_2D computes a line perpendicular to a line and through a point.
#
#  Discussion:
#
#    The explicit form of a line in 2D is:
#
#      ( P1, P2 ) = ( (X1,Y1), (X2,Y2) ).
#
#    The input point P3 should NOT lie on the line (P1,P2).  If it
#    does, then the output value P4 will equal P3.
#
#    P1-----P4-----------P2
#            |
#            |
#           P3
#
#    P4 is also the nearest point on the line (P1,P2) to the point P3.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    25 October 2015
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, real P1(2), P2(2), two points on the line.
#
#    Input, real P3(2), a point (presumably not on the
#    line (P1,P2)), through which the perpendicular must pass.
#
#    Output, real P4(2), a point on the line (P1,P2),
#    such that the line (P3,P4) is perpendicular to the line (P1,P2).
#
#    Output, logical FLAG, is TRUE if the value could not be computed.
#
  import numpy as np

  bot = ( p2[0] - p1[0] ) ** 2 + ( p2[1] - p1[1] ) **2

  p4 = np.zeros ( 2 )

  if ( bot == 0.0 ):
    p4[0] = float ( 'inf' )
    p4[1] = float ( 'inf' )
    flag = 1
#
#  (P3-P1) dot (P2-P1) = Norm(P3-P1) * Norm(P2-P1) * Cos(Theta).
#
#  (P3-P1) dot (P2-P1) / Norm(P3-P1)**2 = normalized coordinate T
#  of the projection of (P3-P1) onto (P2-P1).
#
  t = ( ( p1[0] - p3[0] ) * ( p1[0] - p2[0] ) \
      + ( p1[1] - p3[1] ) * ( p1[1] - p2[1] ) ) / bot

  p4[0] = p1[0] + t * ( p2[0] - p1[0] )
  p4[1] = p1[1] + t * ( p2[1] - p1[1] )

  flag = 0

  return p4, flag

#! /usr/bin/env python
#
def r8mat_solve ( n, nrhs, a ):

#*****************************************************************************80
#
## R8MAT_SOLVE uses Gauss-Jordan elimination to solve an N by N linear system.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    17 October 2015
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer N, the order of the matrix.
#
#    Input, integer NRHS, the number of right hand sides.  NRHS
#    must be at least 0.
#
#    Input, real A(N,N+NRHS), contains in rows and
#    columns 1 to N the coefficient matrix, and in columns N+1 through
#    N+NRHS, the right hand sides.
#
#    Output, real A(N,N+NRHS), the coefficient matrix
#    area has been destroyed, while the right hand sides have
#    been overwritten with the corresponding solutions.
#
#    Output, integer INFO, singularity flag.
#    0, the matrix was not singular, the solutions were computed;
#    J, factorization failed on step J, and the solutions could not
#    be computed.
#
  info = 0

  for j in range ( 0, n ):
#
#  Choose a pivot row IPIVOT.
#
    ipivot = j
    apivot = a[j,j]

    for i in range ( j + 1, n ):
      if ( abs ( apivot ) < abs ( a[i,j] ) ):
        apivot = a[i,j]
        ipivot = i

    if ( apivot == 0.0 ):
      info = j
      return a, info
#
#  Interchange.
#
    for k in range ( 0, n + nrhs ):
      temp        = a[ipivot,k]
      a[ipivot,k] = a[j,k]
      a[j,k]      = temp
#
#  A(J,J) becomes 1.
#
    a[j,j] = 1.0
    for k in range ( j + 1, n + nrhs ):
      a[j,k] = a[j,k] / apivot;
#
#  A(I,J) becomes 0.
#
    for i in range ( 0, n ):

      if ( i != j ):

        factor = a[i,j]
        a[i,j] = 0.0
        for k in range ( j + 1, n + nrhs ):
          a[i,k] = a[i,k] - factor * a[j,k]

  return a, info

def r8mat_solve_test ( ):

#*****************************************************************************80
#
## R8MAT_SOLVE_TEST tests R8MAT_SOLVE.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    29 February 2016
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform
  from r8mat_print import r8mat_print

  n = 3
  rhs_num = 2
#
#  Each row of this definition is a COLUMN of the matrix.
#
  a = np.array ( [ \
    [ 1.0, 2.0, 3.0, 14.0,  7.0 ], \
    [ 4.0, 5.0, 6.0, 32.0, 16.0 ], \
    [ 7.0, 8.0, 0.0, 23.0,  7.0 ] ] )

  print ( '' )
  print ( 'R8MAT_SOLVE_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8MAT_SOLVE solves linear systems.' )
#
#  Print out the matrix to be inverted.
#
  r8mat_print ( n, n + rhs_num, a, '  The linear system:' )
#
#  Solve the systems.
#
  a, info = r8mat_solve ( n, rhs_num, a )
 
  if ( info != 0 ):
    print ( '' )
    print ( '  The input matrix was singular.' )
    print ( '  The solutions could not be computed.' )
    return

  r8mat_print ( n, n + rhs_num, a, '  Factored matrix and solutions:' )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8MAT_SOLVE_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  r8mat_solve_test ( )
  timestamp ( )



def lines_imp_int_2d ( a1, b1, c1, a2, b2, c2 ):

#*****************************************************************************80
#
## LINES_IMP_INT_2D determines where two implicit lines intersect in 2D.
#
#  Discussion:
#
#    The implicit form of a line in 2D is:
#
#      A * X + B * Y + C = 0
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    04 December 2010
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, real A1, B1, C1, define the first line.
#    At least one of A1 and B1 must be nonzero.
#
#    Input, real A2, B2, C2, define the second line.
#    At least one of A2 and B2 must be nonzero.
#
#    Output, integer IVAL, reports on the intersection.
#    -1, both A1 and B1 were zero.
#    -2, both A2 and B2 were zero.
#     0, no intersection, the lines are parallel.
#     1, one intersection point, returned in X, Y.
#     2, infinitely many intersections, the lines are identical.
#
#    Output, real P(2,1), if IVAL = 1, then P is
#    the intersection point.  if IVAL = 2, then P is one of the
#    points of intersection.  Otherwise, P = [].
#
  import numpy as np

  p = np.zeros ( 2 )
#
#  Refuse to handle degenerate lines.
#
  if ( a1 == 0.0 and b1 == 0.0 ):
    ival = -1
    return ival, p
  elif ( a2 == 0.0 and b2 == 0.0 ):
    ival = -2
    return ival, p
#
#  Set up and solve a linear system.
#
  a = np.zeros ( [ 2, 3 ] )

  a[0,0] = a1
  a[0,1] = b1
  a[0,2] = - c1

  a[1,0] = a2
  a[1,1] = b2
  a[1,2] = - c2

  a, info = r8mat_solve ( 2, 1, a )
#
#  If the inverse exists, then the lines intersect at the solution point.
#
  if ( info == 0 ):

    ival = 1;
    p[0] = a[0,2]
    p[1] = a[1,2]
#
#  If the inverse does not exist, then the lines are parallel
#  or coincident.  Check for parallelism by seeing if the
#  C entries are in the same ratio as the A or B entries.
#
  else:

    ival = 0

    if ( a1 == 0.0 ):
      if ( b2 * c1 == c2 * b1 ):
        ival = 2
        p[0] = 0.0
        p[1] = - c1 / b1
    else:
      if ( a2 * c1 == c2 * a1 ):
        ival = 2
        if ( abs ( a1 ) < abs ( b1 ) ):
          p[0] = 0.0
          p[1] = - c1 / b1
        else:
          p[0] = - c1 / a1
          p[1] = 0.0

  return ival, p



def line_exp2imp_2d ( p1, p2 ):

#*****************************************************************************80
#
## LINE_EXP2IMP_2D converts an explicit line to implicit form in 2D.
#
#  Discussion:
#
#    The explicit form of a line in 2D is:
#
#      ( P1, P2 ) = ( (X1,Y1), (X2,Y2) ).
#
#    The implicit form of a line in 2D is:
#
#      A * X + B * Y + C = 0
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    25 October 2015
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, real P1(2,1), P2(2,1), two points on the line.
#
#    Output, real A, B, C, the implicit form of the line.
#
  from sys import exit
#
#  Take care of degenerate cases.
#
  if ( p1[0] == p2[0] and p1[1] == p2[1] ):
    print ( '' )
    print ( 'LINE_EXP2IMP_2D - Fatal error!' )
    print ( '  P1 = P2' )
    print ( '  P1 = %g  %g' % ( p1[0], p1[1] ) )
    print ( '  P2 = %g  %g' % ( p2[0], p2[1] ) )
    exit ( 'LINE_EXP2IMP_2D - Fatal error!' )

  a = p2[1] - p1[1]
  b = p1[0] - p2[0]
  c = p2[0] * p1[1] - p1[0] * p2[1]

  norm = a * a + b * b + c * c

  if ( 0.0 < norm ):
    a = a / norm
    b = b / norm
    c = c / norm

  if ( a < 0.0 ):
    a = -a
    b = -b
    c = -c

  return a, b, c



#! /usr/bin/env python
#
def line_exp2imp_2d ( p1, p2 ):

#*****************************************************************************80
#
## LINE_EXP2IMP_2D converts an explicit line to implicit form in 2D.
#
#  Discussion:
#
#    The explicit form of a line in 2D is:
#
#      ( P1, P2 ) = ( (X1,Y1), (X2,Y2) ).
#
#    The implicit form of a line in 2D is:
#
#      A * X + B * Y + C = 0
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    25 October 2015
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, real P1(2,1), P2(2,1), two points on the line.
#
#    Output, real A, B, C, the implicit form of the line.
#
  from sys import exit
#
#  Take care of degenerate cases.
#
  if ( p1[0] == p2[0] and p1[1] == p2[1] ):
    print ( '' )
    print ( 'LINE_EXP2IMP_2D - Fatal error!' )
    print ( '  P1 = P2' )
    print ( '  P1 = %g  %g' % ( p1[0], p1[1] ) )
    print ( '  P2 = %g  %g' % ( p2[0], p2[1] ) )
    exit ( 'LINE_EXP2IMP_2D - Fatal error!' )

  a = p2[1] - p1[1]
  b = p1[0] - p2[0]
  c = p2[0] * p1[1] - p1[0] * p2[1]

  norm = a * a + b * b + c * c

  if ( 0.0 < norm ):
    a = a / norm
    b = b / norm
    c = c / norm

  if ( a < 0.0 ):
    a = -a
    b = -b
    c = -c

  return a, b, c



def lines_exp_int_2d ( p1, p2, q1, q2 ):

#*****************************************************************************80
#
## LINES_EXP_INT_2D determines where two explicit lines intersect in 2D.
#
#  Discussion:
#
#    The explicit form of a line in 2D is:
#
#      the line through the points P1, P2.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    25 October 2015
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, real P1(2,1), P2(2,1), two points on the first line.
#
#    Input, real Q1(2,1), Q2(2,1), two points on the second line.
#
#    Output, integer IVAL, reports on the intersection:
#    0, no intersection, the lines may be parallel or degenerate.
#    1, one intersection point, returned in P.
#    2, infinitely many intersections, the lines are identical.
#
#    Output, real P(2,1), if IVAl = 1, P is
#    the intersection point.  Otherwise, P = 0.
#
  import numpy as np

  ival = False
  p = np.zeros ( 2 )
#
#  Check whether either line is a point.
#
  if ( p1[0] == p2[0] and p1[1] == p2[1] ):
    point_1 = True
  else:
    point_1 = False

  if ( q1[0] == q2[0] and q1[1] == q2[1] ):
    point_2 = True
  else:
    point_2 = False
#
#  Convert the lines to ABC format.
#
  if ( not point_1 ):
    a1, b1, c1 = line_exp2imp_2d ( p1, p2 )

  if ( not point_2 ):
    a2, b2, c2 = line_exp2imp_2d ( q1, q2 )
#
#  Search for intersection of the lines.
#
  if ( point_1 and point_2 ):
    if ( p1[0] == q1[0] and p1[1] == q1[1] ):
      ival = True
      p[0] = p1[0]
      p[1] = p1[1]
  elif ( point_1 ):
    if ( a2 * p1[0] + b2 * p1[1] == c2 ):
      ival = True
      p[0] = p1[0]
      p[1] = p1[1]
  elif ( point_2 ):
    if ( a1 * q1[0] + b1 * q1[1] == c1 ):
      ival = True
      p[0] = q1[0]
      p[1] = q1[1]
  else:
    ival, p = lines_imp_int_2d ( a1, b1, c1, a2, b2, c2 )

  return ival, p


def triangle_orthocenter ( t ):

#*****************************************************************************80
#
## TRIANGLE_ORTHOCENTER computes the orthocenter of a triangle in 2D.
#
#  Discussion:
#
#    The orthocenter is defined as the intersection of the three altitudes
#    of a triangle.
#
#    An altitude of a triangle is the line through a vertex of the triangle
#    and perpendicular to the opposite side.
#
#    In geometry, the orthocenter of a triangle is often symbolized by "H".
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    25 October 2015
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    Adrian Bowyer and John Woodwark,
#    A Programmer's Geometry,
#    Butterworths, 1983.
#
#  Parameters:
#
#    Input, real T(2,3), the triangle vertices.
#
#    Output, real CENTER(2,1), the orthocenter of the triangle.
#
#    Output, logical FLAG, is TRUE if the value could not be computed.
#
  import numpy as np

  p1 = np.zeros ( 2 )
  p1[0] = t[0,0]
  p1[1] = t[1,0]
  p2 = np.zeros ( 2 )
  p2[0] = t[0,1]
  p2[1] = t[1,1]
  p3 = np.zeros ( 2 )
  p3[0] = t[0,2]
  p3[1] = t[1,2]
  center = np.zeros ( 2 )
#
#  Determine a point P23 common to the line (P2,P3) and
#  its perpendicular through P1.
#
  p23, flag = line_exp_perp_2d ( p2, p3, p1 )

  if ( flag ):
    center[0] = float ( 'inf' )
    center[1] = float ( 'inf' )
    return center, flag
#
#  Determine a point P31 common to the line (P3,P1) and
#  its perpendicular through P2.
#
  p31, flag = line_exp_perp_2d ( p3, p1, p2 )

  if ( flag ):
    center[0] = float ( 'inf' )
    center[1] = float ( 'inf' )
    return center, flag
#
#  Determine CENTER, the intersection of the lines (P1,P23) and (P2,P31).
#
  ival, center = lines_exp_int_2d ( p1, p23, p2, p31 )

  if ( ival != 1 ):
    flag = 1
    center[0] = float ( 'inf' )
    center[1] = float ( 'inf' )
    return center, flag

  return center, flag