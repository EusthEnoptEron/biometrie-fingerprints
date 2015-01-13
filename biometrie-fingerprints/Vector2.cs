using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Biometrie
{
    public struct Vector2
    {
        public double x;
        public double y;

        public Vector2(double p1, double p2)
        {
            this.x = p1;
            this.y = p2;
        }

        public static Vector2 operator +(Vector2 v1, Vector2 v2)
        {
            return new Vector2(v1.x + v2.x, v1.y + v2.y);
        }

        public bool Empty
        {
            get { return x == 0 && y == 0; }
        }
    }
}
