package reflection;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

public class TestRunner {

  Class<?> cls;
  Object obj;

  public TestRunner(String className) {
    try {
      cls = Class.forName(className);
    } catch (ClassNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    try {
      obj = cls.newInstance();
    } catch (InstantiationException | IllegalAccessException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }

  public List<Method> getTests() {
    Method[] methods = cls.getMethods();
    List<Method> testMethods = new ArrayList<>();
    for (Method method : methods) {
      Test testAnnotation = method.getAnnotation(Test.class);
      if (testAnnotation != null) {
        Type type = method.getGenericReturnType();
        if (!type.getTypeName().equals("boolean")) {
          throw new RuntimeException("Invalid return type " + type.getTypeName());
        }
        Type[] arguments = method.getGenericParameterTypes();
        if (arguments.length != 0) {
          throw new RuntimeException("Invalid arguments " + arguments.toString());
        }
        if (!Modifier.isPublic(method.getModifiers())) {
          // throw new Exception("Invalid modifier type " +
          // method.getModifiers());
          System.out.println("Method is not public " + method.getName());
          continue;
        }
        testMethods.add(method);
      }
    }
    return testMethods;
  }

  public List<Method> getBefore() {
    Method[] methods = cls.getMethods();
    List<Method> beforeMethods = new ArrayList<>();
    for (Method method : methods) {
      Before beforeAnno = method.getAnnotation(Before.class);
      if (beforeAnno != null) {
        Type type = method.getGenericReturnType();
        if (!type.getTypeName().equals("void")) {
          throw new RuntimeException("Invalid return type " + type.getTypeName());
        }
        Type[] arguments = method.getGenericParameterTypes();
        if (arguments.length != 0) {
          throw new RuntimeException("Invalid arguments " + arguments.toString());
        }
        if (!Modifier.isPublic(method.getModifiers())) {
          // throw new Exception("Invalid modifier type " +
          // method.getModifiers());
          System.out.println("Method is not public " + method.getName());
          continue;
        }
        beforeMethods.add(method);
      }
    }
    return beforeMethods;
  }

  public List<Method> getAfter() {
    Method[] methods = cls.getMethods();
    List<Method> beforeMethods = new ArrayList<>();
    for (Method method : methods) {
      After beforeAnno = method.getAnnotation(After.class);
      if (beforeAnno != null) {
        Type type = method.getGenericReturnType();
        if (!type.getTypeName().equals("void")) {
          throw new RuntimeException("Invalid return type " + type.getTypeName());
        }
        Type[] arguments = method.getGenericParameterTypes();
        if (arguments.length != 0) {
          throw new RuntimeException("Invalid arguments " + arguments.toString());
        }
        if (!Modifier.isPublic(method.getModifiers())) {
          // throw new Exception("Invalid modifier type " +
          // method.getModifiers());
          System.out.println("Method is not public " + method.getName());
          continue;
        }
        beforeMethods.add(method);
      }
    }
    return beforeMethods;
  }

  public static void main(String[] args) throws IllegalAccessException, IllegalArgumentException,
      InvocationTargetException {
    TestRunner runner = new TestRunner("reflection.MyTest");
    List<Method> before = runner.getBefore();
    List<Method> tests = runner.getTests();
    List<Method> after = runner.getAfter();
    System.out.println("running tests");
    int passed = 0;
    for (Method test : tests) {
      System.out.println("run before");
      for (Method bef : before) {
        bef.invoke(runner.obj);
      }
      Object result = test.invoke(runner.obj);
      boolean resB = (Boolean) result;
      if (resB == true) {
        passed++;
        System.out.println(test.getName() + " " + "passed");
      } else {
        System.out.println(test.getName() + " " + "failed");
      }
      for (Method aft : after) {
        aft.invoke(runner.obj);
      }
    }

  }
}
