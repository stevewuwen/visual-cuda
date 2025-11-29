# CUDA 可视化工具

本项目是一个基于 Web 的交互式工具，旨在为常见的 CUDA C++ 操作提供可视化。它的目标是帮助开发者和学习者以清晰直观的视觉形式，理解复杂的并行计算概念，例如线程层级、内存访问模式（合并与非合并访问）以及核函数执行流程。

## ✨ 特性

-   **交互式模拟**: 一步步执行 CUDA 核函数，观察每个线程的状态变化。
-   **动态加载**: 自动检测并加载 `src/components` 目录下的所有可视化组件。
-   **清晰的UI**: 使用 Tailwind CSS 构建的现代化、简洁的用户界面。
-   **易于扩展**: 只需创建新文件并遵循简单约定，即可轻松添加新的可视化。

## 🛠️ 技术栈

-   **框架**: [React](https://react.dev/)
-   **语言**: [TypeScript](https://www.typescriptlang.org/)
-   **构建工具**: [Vite](https://vitejs.dev/)
-   **UI 样式**: [Tailwind CSS](https://tailwindcss.com/)
-   **路由**: [React Router](https://reactrouter.com/)
-   **动画**: [Framer Motion](https://www.framer.com/motion/)

## 🚀 本地开发

请遵循以下步骤在本地运行项目：

1.  **安装依赖**:
    打开终端，执行以下命令：
    ```bash
    npm install
    ```

2.  **启动开发服务器**:
    ```bash
    npm run dev
    ```
    项目将启动在 `http://localhost:5173` (如果端口被占用，可能会是其他端口)。

## 🎨 如何添加新的可视化

你可以轻松地为项目贡献新的可视化页面。

1.  在 `src/components/` 目录下创建一个新的 `.tsx` 文件 (例如, `my-new-vis.tsx`)。

2.  在文件中，创建你的 React 组件并将其作为 **默认导出** (default export)。

3.  同时，你 **必须** 在该文件中添加一个名为 `metadata` 的 **命名导出** (named export)。这个对象包含了显示在主页上的信息，其结构如下：

    ```typescript
    import MyImage from '/path/to/your/image.svg'; // 引入你的图片资源

    // 定义页面元数据
    export const metadata = {
      name: '我的新可视化', // 页面标题
      description: '这里是关于这个可视化页面的简短描述。', // 页面简介
      image: MyImage, // 用于展示的图片
    };

    // 你的可视化组件
    const MyNewVis = () => {
      // ...组件逻辑
    };

    // 将组件作为默认导出
    export default MyNewVis;
    ```

完成以上步骤后，主页将自动发现并加载你的新组件，无需任何额外配置。
