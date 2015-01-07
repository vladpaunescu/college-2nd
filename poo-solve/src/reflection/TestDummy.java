package reflection;

import java.lang.reflect.Field;
import java.lang.reflect.Method;

public class TestDummy {
  public static void main(String[] args) {

    // no paramater
    Class noparams[] = {};

    // String parameter
    Class[] paramString = new Class[1];
    paramString[0] = String.class;

    // int parameter
    Class[] paramInt = new Class[1];
    paramInt[0] = Integer.TYPE;

    try {
      // load the AppTest at runtime
      Class cls = Class.forName("reflection.ReflectionDummy");
      Object obj = cls.newInstance();
      Field[] fields = cls.getDeclaredFields();
      for (Field f : fields) {
        f.setAccessible(true);
        f.set(obj, 10);
      }

      // call the show method
      Method method = cls.getDeclaredMethod("show", noparams);
      method.invoke(obj);

    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
}
