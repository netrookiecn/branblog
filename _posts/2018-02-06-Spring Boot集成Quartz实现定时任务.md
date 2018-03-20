---
layout: post  
title: "Spring Boot集成Quartz实现定时任务"  
date: 2018-02-06  
excerpt: "实现Spring Boot集成Quartz并使得项目启动时执行某些定时任务"
tags: [Spring Boot,Quartz]
comments: true
---

> 产生这个需求的主要原因是写了个简单的爬取网易新闻的方法，但是总不能一直自己手动执行吧，于是了解到了定时任务以及Quartz这个强大的框架，能够设定执行时间间隔和和何时停止等等。总之框架就是非常的优秀。

本文主要介绍使用Spring Boot项目集成Quartz的过程及遇到的一些问题。  

第一，依赖的两个重要包。


```
<!-- https://mvnrepository.com/artifact/org.quartz-scheduler/quartz -->
        <dependency>
            <groupId>org.quartz-scheduler</groupId>
            <artifactId>quartz</artifactId>
            <version>2.3.0</version>
        </dependency>
<!-- https://mvnrepository.com/artifact/org.springframework/spring-context-support -->
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-context-support</artifactId>
        </dependency>
```

其中，spring-context-support的作用为，quartz在运行任务时，是在spring容器里面吧，会用到其中的一些bean也就是类的实例吧，这个包就对这个需求实现了支持。

第二，以我自身需求为例，我想要定时执行的任务是crawlBasicNewsInfo()这个方法，具体方法内容就不给出了。这个任务方法的类实现Job类，并覆写execute方法，如下代码所示：

``` 
public class CrawlNewsJob implements Job {
 @Autowired
    private NewsRepository newsRepository;

    @Autowired
    private NewsmoduleRepository newsmoduleRepository;

    private static Logger logger = Logger.getLogger(CrawlNewsJob.class);

    @Override
    public void execute(JobExecutionContext context){
        crawlBasicNewsInfo();
    }
```
    
第三，实现对job实例注入spring容器里面的一些bean的任务，首先是覆写框架创建JobInstance的方式，，定制我们自己的SpringJobFactory。不这么做的话，肯定会空指针了。


```
@Component
public class SpringJobFactory extends AdaptableJobFactory{
    @Autowired
    private AutowireCapableBeanFactory capableBeanFactory;

    @Override
    protected Object createJobInstance(TriggerFiredBundle bundle) throws Exception{
        //调用父类获取jobinstance
        Object jobinstance = super.createJobInstance(bundle);
        //此处给job实例注入spring容器中的bean
        capableBeanFactory.autowireBean(jobinstance);
        return jobinstance;
    }
}
```

接着通过@Configuration、@Bean等注解进行配置，这里就是Spring Boot的方便之处了，不需要xml去配置了。值得注意的是，对于注入的Schduler，必须要加入@Primary确保注入的是我们这个Bean，而不是框架中的，不加会出现Field scheduler in RecommenderApplication required a single bean, but 2 were found报错，如图所示。

![](/Users/jiangxl/Documents/公众号/图片/@primary错误.png)

```
@Configuration
public class QuartzConfig {
    @Autowired
    private SpringJobFactory springJobFactory;

    @Bean(name = "SchedulerFactoryBean")
    public SchedulerFactoryBean schedulerFactoryBean(){
        SchedulerFactoryBean schedulerFactoryBean = new SchedulerFactoryBean();
        schedulerFactoryBean.setJobFactory(springJobFactory);
        return schedulerFactoryBean;
    }
    @Bean(name = "Scheduler")
    @Primary
    public Scheduler scheduler(){
        return schedulerFactoryBean().getScheduler();
    }
}
```

第四，实现了上述功能就可以实现定时任务了，但是如果现在需要项目启动时就一直运行怎么办？项目启动时，需要跟Spring Boot一起运行的话，Spring Boot提供了两种自启动方式，一个是CommandLineRunner，一个是ApplicationRunner两个接口，实现它的run方法，把你的定时任务放在其中就好啦。比如我的下面代码所示。

```
@SpringBootApplication
public class RecommenderApplication implements ApplicationRunner{
    private static Logger logger = Logger.getLogger(CrawlNewsJob.class);
    @Autowired
    private  Scheduler scheduler;

    public static void main(String[] args) throws SchedulerException,FileLockInterruptionException {
        SpringApplication.run(RecommenderApplication.class, args);
    }

    @Override
    public void run(ApplicationArguments args) throws Exception{
        try {
            //Scheduler scheduler = StdSchedulerFactory.getDefaultScheduler();
            scheduler.start();
            System.out.println("Quartz Start !");
            //具体任务
            JobDetail job = JobBuilder.newJob(CrawlNewsJob.class).withIdentity("job1","group1").build();

            //触发器
            SimpleScheduleBuilder simpleScheduleBuilder = SimpleScheduleBuilder.simpleSchedule().withIntervalInMinutes(60).repeatForever();
            Trigger trigger = TriggerBuilder.newTrigger().withIdentity("trigger1","group1").startNow().withSchedule(simpleScheduleBuilder).build();

            scheduler.scheduleJob(job,trigger);

            //睡眠
            TimeUnit.MINUTES.sleep(360);
            scheduler.shutdown();
            System.out.println("scheduler shutdown ! ");
        }catch(Exception e){e.printStackTrace();}
    }
}
```

实现Spring Boot集成Quartz并使得项目启动时执行某些定时任务，大致就是上面这些内容了。开始你的表演吧。

