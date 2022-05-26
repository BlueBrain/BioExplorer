create table etype
(
    guid        integer not null
        constraint etypes_pk
            primary key,
    description varchar
)
    tablespace bioexplorer_ts;

alter table etype
    owner to bioexplorer;

create table mesh
(
    guid     integer not null
        constraint mesh_pk
            primary key,
    vertices bytea   not null,
    indices  bytea   not null,
    normals  bytea,
    colors   bytea
)
    tablespace bioexplorer_ts;

alter table mesh
    owner to bioexplorer;

create unique index mesh_guid_uindex
    on mesh (guid)
    tablespace bioexplorer_ts;

create table region
(
    guid              integer                                     not null
        constraint region_pk
            primary key,
    code              varchar                                     not null,
    description       varchar,
    parent_guid       integer default 0                           not null,
    level             integer default 0                           not null,
    atlas_guid        integer default 0                           not null,
    ontology_guid     integer default 0                           not null,
    color_hex_triplet varchar default 'FFFFFF'::character varying not null,
    graph_order       integer default 0                           not null,
    hemisphere_guid   integer default 0                           not null
)
    tablespace bioexplorer_ts;

alter table region
    owner to bioexplorer;

create table cell
(
    guid                 integer                    not null
        constraint cells_pk
            primary key,
    cell_type_guid       integer          default 0 not null,
    region_guid          integer                    not null
        constraint cell_region_guid_fk
            references region
            on update cascade on delete cascade,
    electrical_type_guid integer          default 0 not null
        constraint cell_etype_guid_fk
            references etype
            on update cascade on delete cascade,
    x                    double precision           not null,
    y                    double precision           not null,
    z                    double precision           not null,
    rotation_x           double precision default 0 not null,
    rotation_y           double precision default 0 not null,
    rotation_z           double precision default 0 not null,
    rotation_w           double precision default 1 not null
)
    tablespace bioexplorer_ts;

alter table cell
    owner to bioexplorer;

create index cell_region_guid_index
    on cell (region_guid)
    tablespace bioexplorer_ts;

