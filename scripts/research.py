"""Run/resume the same persistent workflow as the UI, including manual selection."""
import argparse
from research_agent.checkpoints import list_runs
from research_agent.deletion import preview_deletion, delete_run
from research_agent.orchestrator import run_research


def main():
    parser = argparse.ArgumentParser(description='PaperScout 持久化调研')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('question')
    run.add_argument('--visual-pages', type=int, default=3)
    run.add_argument('--approve-papers', action='store_true')
    resume = commands.add_parser('resume')
    resume.add_argument('run_id')
    resume.add_argument('--papers', nargs='*', default=None, help='确认候选 arXiv ID；省略时恢复失败步骤')
    commands.add_parser('list')
    delete = commands.add_parser('delete', help='删除记录、检查点与非共享缓存')
    delete.add_argument('run_id')
    delete.add_argument('--yes', action='store_true', help='确认执行删除；省略时仅预览')
    delete.add_argument('--keep-cache', action='store_true', help='保留论文、PDF 和图片缓存')
    args = parser.parse_args()
    if args.command == 'delete':
        try:
            plan = (delete_run if args.yes else preview_deletion)(args.run_id, clean_cache=not args.keep_cache)
        except (ValueError, OSError) as exc:
            parser.exit(1, str(exc) + '\n')
        print(('已删除' if args.yes else '预览') + '：' + plan.question)
        print(f'缓存文件：{len(plan.files)}；大小：{plan.bytes / 1024**2:.2f} MB；共享保留：{plan.shared_files}')
        for warning in plan.warnings:
            print(warning)
        if not args.yes:
            print('执行删除时添加 --yes；删除后无法恢复。')
        return
    if args.command == 'list':
        for result in list_runs():
            print(result.run_id, result.status, result.question)
        return
    kwargs = {'question': args.question, 'visual_pages': args.visual_pages, 'approve_papers': args.approve_papers} if args.command == 'run' else {
        'run_id': args.run_id, 'resume': True, 'selection': args.papers}
    for event in run_research(**kwargs):
        print(event.message, flush=True)
        if event.stage == 'checkpoint':
            print('运行 ID：', event.data['run_id'], flush=True)
        if event.stage == 'done':
            print('最终状态：', event.data.status, '可恢复：', event.data.resumable)
            if event.data.pending_selection:
                print(event.data.pending_selection)


if __name__ == '__main__':
    main()
