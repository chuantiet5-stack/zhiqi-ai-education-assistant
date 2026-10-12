"""
技能加载器模块

SkillsLoader 负责从 skills 目录加载技能定义，包括：
- 发现所有可用技能（扫描子目录中的 SKILL.md）
- 解析 SKILL.md 的 frontmatter 元数据
- 构建技能摘要供 Agent 了解可用能力
- 加载具体技能的详细指南

技能目录结构示例：
    skills/
    ├── code-review/
    │   └── SKILL.md      # 技能定义文件
    ├── git-helper/
    │   └── SKILL.md
    └── ...

SKILL.md 格式（可选 frontmatter）：
    ---
    name: code-review
    description: 代码审查技能，检查代码质量和潜在问题
    ---

    # 详细指南内容...
"""

import os
from typing import Any

import yaml


class SkillsLoader:
    """
    技能加载器

    扫描 skills 目录，发现并加载所有可用技能。

    设计意图（渐进式披露）：
    - system prompt 只注入技能的名称、路径与一句话描述（轻量索引，
      节省 Token），不加载任何技能全文
    - Agent 判断需要某项技能时，再用 read_file 读取对应的 SKILL.md
      获取完整操作指南，把详细内容的加载权交给模型按需触发

    使用示例：
        loader = SkillsLoader(skills_dir="skills")

        # 获取技能摘要（用于 system prompt）
        summary = loader.build_skills_summary()

        # 加载特定技能的详细内容
        skill_content = loader.load_skill("code-review")

        # 列出所有技能信息
        skills_list = loader.list_skills()
    """

    def __init__(self, skills_dir: str = "skills") -> None:
        """
        Args:
            skills_dir: 技能目录路径，默认 "skills"
        """
        self.skills_dir = skills_dir

    def _parse_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """
        解析 SKILL.md 文件的 frontmatter

        如果文件以 "---\\n" 开头，找到第二个 "---"，提取中间的 YAML 内容。
        用 yaml.safe_load() 解析 YAML 为字典。
        返回 (metadata字典, 去掉frontmatter后的正文)。
        如果没有 frontmatter，返回 (空字典, 原文)。

        Args:
            content: SKILL.md 文件的完整内容

        Returns:
            tuple[dict, str]: (元数据字典, 正文内容)
        """
        # 检查是否以 frontmatter 开头
        # 手工按分隔符截取而非把整个文件交给 YAML 解析：
        # SKILL.md 主体是 Markdown，只有分隔符之间的一小段才是合法 YAML
        if not content.startswith("---\n"):
            return {}, content

        # 找到第二个 "---"
        # 跳过第一个 "---\n"（4个字符）
        remaining = content[4:]

        # 查找第二个 "---" 的位置
        second_delimiter_pos = remaining.find("---\n")
        if second_delimiter_pos == -1:
            # 没有找到结束分隔符，按无 frontmatter 处理
            return {}, content

        # 提取 YAML 内容
        yaml_content = remaining[:second_delimiter_pos]

        # 提取正文（跳过第二个 "---\n"）
        body = remaining[second_delimiter_pos + 4:].strip()

        # 解析 YAML
        try:
            metadata = yaml.safe_load(yaml_content)
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
        except yaml.YAMLError:
            metadata = {}

        return metadata, body

    def build_skills_summary(self) -> str:
        """
        构建技能摘要字符串

        遍历 skills_dir 下的每个子目录，查找 SKILL.md 文件，
        解析 frontmatter 获取 name 和 description，格式化为摘要列表。

        Returns:
            str: 技能摘要字符串，无技能时返回空字符串
        """
        # 检查技能目录是否存在
        if not os.path.isdir(self.skills_dir):
            return ""

        # 收集所有技能
        skills_info: list[tuple[str, str, str]] = []  # (name, path, description)

        # 遍历子目录
        try:
            for entry in os.listdir(self.skills_dir):
                skill_subdir = os.path.join(self.skills_dir, entry)

                # 只处理目录
                if not os.path.isdir(skill_subdir):
                    continue

                # 查找 SKILL.md
                skill_file = os.path.join(skill_subdir, "SKILL.md")
                if not os.path.isfile(skill_file):
                    continue

                # 读取并解析
                try:
                    with open(skill_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    metadata, _ = self._parse_frontmatter(content)

                    # 获取 name 和 description
                    name = metadata.get("name", entry)  # 默认使用目录名
                    description = metadata.get("description", "无描述")

                    # 记录相对路径
                    relative_path = os.path.join(self.skills_dir, entry, "SKILL.md")

                    skills_info.append((name, relative_path, description))

                except Exception:
                    # 跳过无法读取的文件
                    continue

        except Exception:
            return ""

        # 如果没有找到任何技能，返回空字符串
        if not skills_info:
            return ""

        # 构建摘要字符串
        # 摘要中刻意只列出 name/path/description 并引导 Agent 按需
        # 读取 SKILL.md 全文，避免把所有技能的详细指南塞进 system prompt
        lines = [
            "你有以下技能可用。当你需要使用某项技能时，",
            "请先用 read_file 工具读取对应的 SKILL.md 文件获取详细指南。",
            "",
            "可用技能：",
        ]

        for name, path, description in skills_info:
            lines.append(f"- {name} ({path}): {description}")

        return "\n".join(lines)

    def load_skill(self, name: str) -> str | None:
        """
        加载指定技能的详细内容

        在 skills_dir 下查找名为 name 的子目录中的 SKILL.md，
        读取文件内容，去掉 frontmatter 后返回正文。

        Args:
            name: 技能名称（子目录名）

        Returns:
            str | None: 技能正文内容，找不到时返回 None
        """
        # 构建技能文件路径
        skill_file = os.path.join(self.skills_dir, name, "SKILL.md")

        # 检查文件是否存在
        if not os.path.isfile(skill_file):
            return None

        # 读取文件
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return None

        # 解析并返回正文
        # 丢弃 frontmatter：元数据已在技能摘要中出现过，
        # Agent 此处需要的是正文里的详细操作指南
        metadata, body = self._parse_frontmatter(content)
        return body

    def list_skills(self) -> list[dict[str, Any]]:
        """
        返回所有已发现技能的信息列表

        用于调试和管理。

        Returns:
            list[dict]: 技能信息列表，每项包含 name、description、path
        """
        result: list[dict[str, Any]] = []

        # 检查技能目录是否存在
        if not os.path.isdir(self.skills_dir):
            return result

        # 遍历子目录
        try:
            for entry in os.listdir(self.skills_dir):
                skill_subdir = os.path.join(self.skills_dir, entry)

                # 只处理目录
                if not os.path.isdir(skill_subdir):
                    continue

                # 查找 SKILL.md
                skill_file = os.path.join(skill_subdir, "SKILL.md")
                if not os.path.isfile(skill_file):
                    continue

                # 读取并解析
                try:
                    with open(skill_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    metadata, _ = self._parse_frontmatter(content)

                    # 获取信息
                    skill_name = metadata.get("name", entry)
                    description = metadata.get("description", "无描述")
                    path = os.path.join(self.skills_dir, entry, "SKILL.md")

                    result.append({
                        "name": skill_name,
                        "description": description,
                        "path": path,
                    })

                except Exception:
                    continue

        except Exception:
            pass

        return result