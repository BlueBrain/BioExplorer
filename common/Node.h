#ifndef NODE_H
#define NODE_H

#include <common/types.h>

class Node
{
public:
    Node();

    virtual ~Node();
};

typedef std::shared_ptr<Node> NodePtr;
typedef std::map<std::string, NodePtr> NodeMap;

#endif // NODE_H
