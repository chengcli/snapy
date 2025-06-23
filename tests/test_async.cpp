#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

// Simulate some work
void doWork(int id, int delay) {
  std::cout << "Task " << id << " starting.\n";
  std::this_thread::sleep_for(std::chrono::seconds(delay));
  std::cout << "Task " << id << " done.\n";
}

// Task class to represent a task with dependencies
class Task {
 public:
  Task(int id, int delay,
       std::vector<std::shared_future<void>> dependencies = {})
      : id_(id), delay_(delay), dependencies_(dependencies) {}

  // Execute task asynchronously, resolving dependencies first
  std::future<void> execute() {
    return std::async(std::launch::async, [this]() {
      for (auto& dep : dependencies_) {
        dep.wait();  // Wait for all dependencies to complete
      }
      doWork(id_, delay_);
    });
  }

 private:
  int id_;
  int delay_;
  std::vector<std::shared_future<void>> dependencies_;
};

// Task Scheduler to manage tasks
class TaskScheduler {
 public:
  // Add task to scheduler with optional dependencies
  void addTask(int id, int delay,
               const std::vector<std::shared_future<void>>& dependencies = {}) {
    tasks_.emplace_back(id, delay, dependencies);
  }

  // Execute all tasks asynchronously
  void run() {
    std::vector<std::future<void>> futures;
    for (auto& task : tasks_) {
      futures.push_back(task.execute());
    }

    // Wait for all tasks to complete
    for (auto& future : futures) {
      future.get();
    }
  }

 private:
  std::vector<Task> tasks_;
};

int main() {
  TaskScheduler scheduler;

  // Task 1: Independent task
  auto task1 = std::async(std::launch::async, doWork, 1, 2);
  std::shared_future<void> task1Future =
      task1.share();  // Convert to shared_future for dependencies

  // Task 2: Dependent on task 1
  auto task2 = std::async(std::launch::async, doWork, 2, 1);
  std::shared_future<void> task2Future = task2.share();

  // Task 3: Dependent on task 1 and task 2
  scheduler.addTask(3, 3, {task1Future, task2Future});

  // Task 4: Independent
  scheduler.addTask(4, 1);

  // Task 5: Dependent on task 3
  scheduler.addTask(5, 2, {task1Future});

  scheduler.run();

  std::cout << "All tasks finished.\n";
  return 0;
}
