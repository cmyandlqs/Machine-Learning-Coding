#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖库版本查询和requirements.txt生成脚本
用于自动分析Python项目中的依赖库并生成requirements.txt文件
"""

import os
import sys
import re
import subprocess
from pathlib import Path
import importlib.util

class DependencyAnalyzer:
    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.requirements_path = self.project_path / "requirements.txt"
        self.dependencies = {}
        
    def extract_imports_from_file(self, file_path):
        """从Python文件中提取import语句"""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 匹配各种import语句
            patterns = [
                r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # import module
                r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',  # from module import
                r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as',  # import module as
            ]
            
            for line in content.split('\n'):
                line = line.strip()
                for pattern in patterns:
                    match = re.match(pattern, line)
                    if match:
                        module_name = match.group(1)
                        # 处理子模块，只取主模块名
                        main_module = module_name.split('.')[0]
                        imports.add(main_module)
                        
        except Exception as e:
            print(f"警告: 无法读取文件 {file_path}: {e}")
            
        return imports
    
    def scan_python_files(self):
        """扫描项目中的所有Python文件"""
        all_imports = set()
        
        # 查找所有.py文件
        for py_file in self.project_path.rglob("*.py"):
            if py_file.name != __file__:  # 排除当前脚本
                print(f"扫描文件: {py_file}")
                imports = self.extract_imports_from_file(py_file)
                all_imports.update(imports)
                
        return all_imports
    
    def get_package_version(self, package_name):
        """获取包的版本信息"""
        try:
            # 尝试直接导入
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', None)
            
            if version:
                return version
                
            # 如果模块没有__version__属性，尝试使用pip show
            try:
                result = subprocess.run(
                    ['pip', 'show', package_name], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        return line.split(':', 1)[1].strip()
                        
            except subprocess.CalledProcessError:
                pass
                
            return "Unknown"
            
        except ImportError:
            return "Not Installed"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def map_import_to_package(self, import_name):
        """将import名称映射到实际的包名称"""
        # 常见的import名称到包名称的映射
        mapping = {
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn',
            'PIL': 'Pillow',
            'yaml': 'PyYAML',
            'dateutil': 'python-dateutil',
            'requests': 'requests',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib',
            'pandas': 'pandas',
            'tensorflow': 'tensorflow',
            'torch': 'torch',
            'keras': 'keras',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'bokeh': 'bokeh',
            'django': 'django',
            'flask': 'flask',
            'fastapi': 'fastapi',
            'sqlalchemy': 'sqlalchemy',
            'pymongo': 'pymongo',
            'redis': 'redis',
            'celery': 'celery',
            'gunicorn': 'gunicorn',
            'uvicorn': 'uvicorn',
            'pytest': 'pytest',
            'unittest': 'unittest',  # 内置模块
            'os': 'os',  # 内置模块
            'sys': 'sys',  # 内置模块
            're': 're',  # 内置模块
            'json': 'json',  # 内置模块
            'datetime': 'datetime',  # 内置模块
            'pathlib': 'pathlib',  # 内置模块
            'subprocess': 'subprocess',  # 内置模块
            'importlib': 'importlib',  # 内置模块
        }
        
        return mapping.get(import_name, import_name)
    
    def filter_builtin_modules(self, imports):
        """过滤掉Python内置模块"""
        builtin_modules = {
            'os', 'sys', 're', 'json', 'datetime', 'pathlib', 'subprocess', 
            'importlib', 'collections', 'itertools', 'functools', 'operator',
            'math', 'random', 'string', 'io', 'tempfile', 'shutil', 'glob',
            'pickle', 'copy', 'time', 'calendar', 'hashlib', 'base64',
            'urllib', 'http', 'socket', 'threading', 'multiprocessing',
            'queue', 'logging', 'warnings', 'traceback', 'inspect',
            'unittest', 'doctest', 'pdb', 'profile', 'pstats'
        }
        
        return {imp for imp in imports if imp not in builtin_modules}
    
    def analyze_dependencies(self):
        """分析项目依赖"""
        print("=" * 60)
        print("开始分析项目依赖...")
        print("=" * 60)
        
        # 扫描Python文件
        all_imports = self.scan_python_files()
        print(f"\n发现 {len(all_imports)} 个导入模块")
        
        # 过滤内置模块
        external_imports = self.filter_builtin_modules(all_imports)
        print(f"外部依赖模块: {len(external_imports)} 个")
        
        # 获取版本信息
        print("\n查询依赖库版本信息:")
        print("-" * 40)
        
        for import_name in sorted(external_imports):
            package_name = self.map_import_to_package(import_name)
            version = self.get_package_version(import_name)
            
            print(f"{import_name:20} -> {package_name:20} : {version}")
            
            # 只记录已安装的包
            if version != "Not Installed" and version != "Unknown":
                self.dependencies[package_name] = version
                
        return self.dependencies
    
    def generate_requirements_txt(self):
        """生成requirements.txt文件"""
        print(f"\n生成 requirements.txt 文件: {self.requirements_path}")
        print("-" * 40)
        
        try:
            with open(self.requirements_path, 'w', encoding='utf-8') as f:
                f.write("# 自动生成的依赖库文件\n")
                f.write("# 生成时间: " + str(subprocess.run(['date'], capture_output=True, text=True).stdout.strip()) + "\n")
                f.write("# 项目路径: " + str(self.project_path) + "\n\n")
                
                if self.dependencies:
                    for package, version in sorted(self.dependencies.items()):
                        f.write(f"{package}=={version}\n")
                        print(f"✅ {package}=={version}")
                else:
                    f.write("# 未发现外部依赖库\n")
                    print("⚠️  未发现外部依赖库")
                    
            print(f"\n✅ requirements.txt 已生成: {self.requirements_path}")
            
        except Exception as e:
            print(f"❌ 生成 requirements.txt 失败: {e}")
    
    def run_analysis(self):
        """运行完整的依赖分析"""
        try:
            # 分析依赖
            dependencies = self.analyze_dependencies()
            
            # 生成requirements.txt
            self.generate_requirements_txt()
            
            print("\n" + "=" * 60)
            print("依赖分析完成!")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            sys.exit(1)

def main():
    """主函数"""
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    project_path = script_dir / "K_Nearest_Neighbor" / "DEMO-Image_Style_Transfer"
    
    print("KNN图像风格迁移项目 - 依赖分析工具")
    print("=" * 60)
    print(f"项目路径: {project_path}")
    
    if not project_path.exists():
        print(f"❌ 项目路径不存在: {project_path}")
        sys.exit(1)
    
    # 创建分析器并运行
    analyzer = DependencyAnalyzer(project_path)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()