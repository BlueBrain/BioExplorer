create table atlas.region
(
    guid        integer           not null
        constraint region_pk
            primary key,
    code        varchar           not null,
    description varchar,
    parent_guid integer default 0 not null,
    level       integer default 0 not null
)
    tablespace bioexplorer_ts;

alter table atlas.region
    owner to bioexplorer;

create table atlas.etype
(
    guid        integer not null
        constraint etypes_pk
            primary key,
    description varchar
)
    tablespace bioexplorer_ts;

alter table atlas.etype
    owner to bioexplorer;

create table atlas.cell
(
    guid                 integer                    not null
        constraint cells_pk
            primary key,
    region_guid          integer                    not null
        constraint cell_region_guid_fk
            references atlas.region
            on update cascade on delete cascade,
    electrical_type_guid integer          default 0 not null
        constraint cell_etype_guid_fk
            references atlas.etype
            on update cascade on delete cascade,
    x                    double precision           not null,
    y                    double precision           not null,
    z                    double precision           not null,
    rotation_x           double precision default 0 not null,
    rotation_y           double precision default 0 not null,
    rotation_z           double precision default 0 not null,
    rotation_w           double precision default 1 not null,
    cell_type_guid       integer          default 0 not null
)
    tablespace bioexplorer_ts;

alter table atlas.cell
    owner to bioexplorer;

