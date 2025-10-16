from crossref_commons.iteration import iterate_publications_as_json
from .data_archive import data_archive

def doi_catch():
    filter = {'prefix':'10.1016','type': 'journal-article'} # 10.1016是Elsvier前缀
    queries = {'query.bibliographic': {'alloy'}} # 'alloy'表示搜索的关键词
    doilist = []
    for p in iterate_publications_as_json(max_results=100, filter=filter, queries=queries):
        # 1000000
        doilist.append(p['DOI'])
    return doilist

if __name__ == '__main__':
    doilist = doi_catch()
    print(doilist)
    data_archive(doilist,"./demo_data/alloyDOI.xlsx","alloyDOI")
