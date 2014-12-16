import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class TestGeometricForms {

  GeometricForms forms;

  @Test
  public void testIsTriangle() {
    assertTrue(new GeometricForms(Forms.CIRCLE.name()).isCircle());
  }

  @Test
  public void testIsCircle() {
    assertTrue(new GeometricForms(Forms.RECTANGLE.name()).isRectangle());
  }

  @Test
  public void testIsRectangle() {
    assertTrue(new GeometricForms(Forms.TRIANGLE.name()).isTriangle());
  }

}
